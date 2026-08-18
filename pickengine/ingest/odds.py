"""Odds ingestion: The Odds API (the-odds-api.com) live pulls + file import.

Two entry paths normalize into the same OddsSnapshot rows:

1. Live pull (`fetch_live_odds` + `ingest_live_events`): The Odds API v4
   `/sports/baseball_mlb/odds` with markets h2h, totals, spreads (their
   "spreads" is the MLB runline). Events are mapped to our games by team
   names (via a defensive alias table) plus commence time: the official date
   is the commence time's UTC date or the day before (official dates lag UTC,
   never lead), and candidates — including doubleheaders — are disambiguated
   by first pitch proximity.

2. File import (`import_odds_file`): CSV or JSON dumps of historical odds —
   how purchased historical snapshots (The Odds API's paid endpoints) or free
   archived datasets get loaded.

   CSV schema (header required, one row per outcome quote):
       date,home_team,away_team,book,market,outcome,decimal_odds,line,timestamp,is_closing
   - date:         game date, YYYY-MM-DD — MLB's OFFICIAL (local) date, the
                   date every odds archive uses. A late ET night game keeps
                   its ET date even though first pitch is past midnight UTC.
   - home_team /
     away_team:    team names (alias table applies)
   - book:         bookmaker key, e.g. "pinnacle"
   - market:       h2h | totals | runline (aliases: moneyline/ml -> h2h,
                   total/ou -> totals, spread/spreads/run_line -> runline)
   - outcome:      team name for h2h/runline; Over or Under for totals
   - decimal_odds: decimal odds, > 1.0
   - line:         empty for h2h; the total or runline number otherwise
   - timestamp:    capture time, ISO-8601; naive values are treated as UTC
   - is_closing:   true/false (empty = false); mark-closing can also derive it
   - commence_time: OPTIONAL column — the scheduled start (ISO-8601, naive
                   treated as UTC) of the specific game the row refers to.
                   Needed only to attribute doubleheader rows: with it, the
                   row is matched to the pair's game with the nearest first
                   pitch; without it, doubleheader rows are skipped and
                   counted under ambiguous_doubleheader (add the column to
                   your dataset to recover them).

   A JSON file is a list of objects with the same field names.

Unmatched or ambiguous games and duplicate rows are skipped and COUNTED
(reported by the CLI, never silent); malformed values raise with the row
number. INTEGRITY RULE: downstream code (model, backtester) must obtain odds
only through `get_usable_odds`, which enforces captured_at_utc strictly
before first pitch and before the query's as-of time.
"""

import csv
import json
import os
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from pickengine.models import Game, Market, OddsSnapshot, Team

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Variant name (normalized) -> our canonical name (normalized). Extend as new
# variants show up; unmatched teams are counted and reported, never guessed.
TEAM_ALIASES = {
    "oakland athletics": "athletics",
    "oakland as": "athletics",
    "cleveland indians": "cleveland guardians",
    "florida marlins": "miami marlins",
    "tampa bay devil rays": "tampa bay rays",
    "anaheim angels": "los angeles angels",
    "los angeles angels of anaheim": "los angeles angels",
    "la angels": "los angeles angels",
    "la dodgers": "los angeles dodgers",
    "ny mets": "new york mets",
    "ny yankees": "new york yankees",
    "chi cubs": "chicago cubs",
    "chi white sox": "chicago white sox",
    "arizona dbacks": "arizona diamondbacks",
    "arizona d backs": "arizona diamondbacks",
    "washington nats": "washington nationals",
}

MARKET_ALIASES = {
    "h2h": Market.H2H,
    "moneyline": Market.H2H,
    "ml": Market.H2H,
    "totals": Market.TOTALS,
    "total": Market.TOTALS,
    "ou": Market.TOTALS,
    "runline": Market.RUNLINE,
    "run_line": Market.RUNLINE,
    "spread": Market.RUNLINE,
    "spreads": Market.RUNLINE,
}

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no", ""}


def normalize_team_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in name.lower())
    return " ".join(cleaned.split())


class GameIndex:
    """Resolves (team names, official date[, time]) -> Game for one run."""

    # A commence/scheduled-start time signal must land within this window of
    # a candidate's first pitch; further than that is a different game (live
    # path) or an unusable signal (file-import doubleheaders).
    MAX_COMMENCE_DELTA = timedelta(hours=6)

    def __init__(self, session: Session) -> None:
        teams = session.scalars(select(Team)).all()
        if not teams:
            raise RuntimeError("no teams in database — run sync-teams first")
        self.team_id_by_name: dict[str, int] = {normalize_team_name(t.name): t.id for t in teams}
        for alias, canonical in TEAM_ALIASES.items():
            if canonical in self.team_id_by_name:
                self.team_id_by_name[alias] = self.team_id_by_name[canonical]
        self.name_by_team_id = {t.id: t.name for t in teams}

        self.games: dict[tuple[int, int, date], list[Game]] = defaultdict(list)
        for game in session.scalars(select(Game)):
            self.games[(game.home_team_id, game.away_team_id, game.official_date)].append(game)

    def resolve_team(self, name: str) -> int | None:
        return self.team_id_by_name.get(normalize_team_name(name))

    def find_game(
        self, home_team_id: int, away_team_id: int, official_date: date,
        near_time: datetime | None = None,
    ) -> Game | None | str:
        """Exact official-date lookup (file imports, which carry a date).

        Returns the Game, None if no match, or "ambiguous" when a
        doubleheader can't be disambiguated: no `near_time` given, the
        candidates carry no first pitch times to compare it against, or the
        nearest first pitch is further than MAX_COMMENCE_DELTA from
        `near_time` (a time signal that far off — a wrong or local-time
        commence_time — must not silently attribute both halves of a
        doubleheader to one game)."""
        candidates = self.games.get((home_team_id, away_team_id, official_date), [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if near_time is not None:
            timed = [g for g in candidates if g.first_pitch_utc is not None]
            if timed:
                nearest = min(timed, key=lambda g: abs(g.first_pitch_utc - near_time))
                if abs(nearest.first_pitch_utc - near_time) <= self.MAX_COMMENCE_DELTA:
                    return nearest
        return "ambiguous"

    def find_game_by_commence(
        self, home_team_id: int, away_team_id: int, commence: datetime
    ) -> Game | None | str:
        """Live-event lookup: The Odds API only gives a UTC commence time.

        The official date is either the commence time's UTC date or the day
        before (official dates lag UTC dates, never lead them), so candidates
        from both dates are pooled and resolved by first pitch proximity —
        which also disambiguates doubleheaders. A nearest candidate further
        than MAX_COMMENCE_DELTA from the commence time is a different game
        (e.g. the next game of the series when this one is not synced), not a
        match. Returns "ambiguous" only when several candidates all lack a
        first pitch time."""
        candidates = []
        for official in (commence.date() - timedelta(days=1), commence.date()):
            candidates += self.games.get((home_team_id, away_team_id, official), [])
        if not candidates:
            return None
        timed = [g for g in candidates if g.first_pitch_utc is not None]
        if timed:
            nearest = min(timed, key=lambda g: abs(g.first_pitch_utc - commence))
            if abs(nearest.first_pitch_utc - commence) <= self.MAX_COMMENCE_DELTA:
                return nearest
            return None
        return candidates[0] if len(candidates) == 1 else "ambiguous"


def _parse_utc(timestamp: str) -> datetime:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _new_counts() -> dict[str, int]:
    return {
        "inserted": 0, "duplicates": 0, "unmatched_team": 0,
        "unmatched_game": 0, "ambiguous_game": 0, "ambiguous_doubleheader": 0,
        "skipped_market": 0,
    }


class _SnapshotWriter:
    """Buffers snapshot rows and flushes them in one executemany INSERT.

    Dedup rides the uq_odds_snapshot_key unique index via ON CONFLICT DO
    NOTHING, so re-imports never load existing keys into memory; conflicted
    rows (within the batch or against existing rows) count as duplicates via
    the statement's cumulative rowcount. One Core executemany instead of one
    INSERT round trip per quote keeps large historical imports fast.
    Validation stays per row (in `add`) so file importers can still attribute
    errors to a row number.
    """

    def __init__(self, session: Session, counts: dict[str, int]) -> None:
        self.session = session
        self.counts = counts
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        game_id: int, book: str, market: Market, outcome_label: str,
        decimal_odds: float, line_value: float | None, captured_at: datetime,
        is_closing: bool,
    ) -> None:
        if decimal_odds <= 1.0:
            raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
        self.rows.append(
            {
                "game_id": game_id, "book": book, "market": market,
                "outcome_label": outcome_label, "decimal_odds": decimal_odds,
                "line_value": line_value, "captured_at_utc": captured_at,
                "is_closing": is_closing,
            }
        )

    def flush(self) -> None:
        if not self.rows:
            return
        result = self.session.connection().execute(
            sqlite_insert(OddsSnapshot).on_conflict_do_nothing(), self.rows
        )
        self.counts["inserted"] += result.rowcount
        self.counts["duplicates"] += len(self.rows) - result.rowcount
        self.rows = []


# --- Path 1: live pull from The Odds API -----------------------------------


def fetch_live_odds(api_key: str, regions: str = "us,eu") -> list[dict[str, Any]]:
    """Fetch current MLB odds (h2h, totals, spreads) from The Odds API v4."""
    response = httpx.get(
        f"{ODDS_API_BASE}/sports/baseball_mlb/odds",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "h2h,totals,spreads",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def ingest_live_events(
    session: Session, events: list[dict[str, Any]], captured_at: datetime
) -> dict[str, int]:
    """Insert snapshots from Odds API event payloads. Returns counts."""
    index = GameIndex(session)
    counts = _new_counts()
    writer = _SnapshotWriter(session, counts)

    for event in events:
        home_id = index.resolve_team(event["home_team"])
        away_id = index.resolve_team(event["away_team"])
        if home_id is None or away_id is None:
            counts["unmatched_team"] += 1
            continue
        commence = _parse_utc(event["commence_time"])
        game = index.find_game_by_commence(home_id, away_id, commence)
        if game is None:
            counts["unmatched_game"] += 1
            continue
        if game == "ambiguous":
            counts["ambiguous_game"] += 1
            continue

        for bookmaker in event["bookmakers"]:
            for market_raw in bookmaker["markets"]:
                market = MARKET_ALIASES.get(market_raw["key"])
                if market is None:
                    counts["skipped_market"] += 1
                    continue
                for outcome in market_raw["outcomes"]:
                    label = _resolve_outcome_label(index, market, outcome["name"])
                    if label is None:
                        counts["unmatched_team"] += 1
                        continue
                    writer.add(
                        game_id=game.id, book=bookmaker["key"], market=market,
                        outcome_label=label, decimal_odds=float(outcome["price"]),
                        line_value=outcome.get("point"), captured_at=captured_at,
                        is_closing=False,
                    )
    writer.flush()
    return counts


def _resolve_outcome_label(index: GameIndex, market: Market, name: str) -> str | None:
    """Canonical outcome label: our Team.name for h2h/runline, Over/Under for totals."""
    if market is Market.TOTALS:
        if name.lower() not in ("over", "under"):
            raise ValueError(f"unexpected totals outcome {name!r}")
        return name.capitalize()
    team_id = index.resolve_team(name)
    return index.name_by_team_id[team_id] if team_id is not None else None


# --- Path 2: CSV / JSON file import -----------------------------------------


def import_odds_file(session: Session, path: str | Path) -> dict[str, int]:
    """Import a CSV or JSON odds dump (schema in module docstring)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported odds file type: {path.suffix!r} (expected .csv or .json)")

    index = GameIndex(session)
    counts = _new_counts()
    writer = _SnapshotWriter(session, counts)
    for line_no, row in enumerate(rows, start=2 if path.suffix.lower() == ".csv" else 1):
        try:
            _ingest_row(index, counts, writer, row)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path.name} row {line_no}: {exc}") from exc
    writer.flush()
    return counts


def _ingest_row(
    index: GameIndex, counts: dict[str, int], writer: _SnapshotWriter,
    row: dict[str, Any],
) -> None:
    market_key = str(row["market"]).strip().lower()
    market = MARKET_ALIASES.get(market_key)
    if market is None:
        raise ValueError(f"unknown market {row['market']!r}")

    home_id = index.resolve_team(str(row["home_team"]))
    away_id = index.resolve_team(str(row["away_team"]))
    if home_id is None or away_id is None:
        counts["unmatched_team"] += 1
        return
    commence_raw = row.get("commence_time")
    near_time = None if commence_raw in (None, "") else _parse_utc(str(commence_raw))
    game = index.find_game(
        home_id, away_id, date.fromisoformat(str(row["date"])), near_time=near_time
    )
    if game is None:
        counts["unmatched_game"] += 1
        return
    if game == "ambiguous":
        if near_time is None:
            # Doubleheader with date-only granularity: cannot safely
            # attribute. Recoverable — add commence_time to the dataset.
            counts["ambiguous_doubleheader"] += 1
        else:
            # Had a time signal but the candidates have no first pitch
            # times to compare against, or none within MAX_COMMENCE_DELTA.
            counts["ambiguous_game"] += 1
        return

    label = _resolve_outcome_label(index, market, str(row["outcome"]))
    if label is None:
        counts["unmatched_team"] += 1
        return

    line_raw = row.get("line")
    line_value = None if line_raw in (None, "") else float(line_raw)
    closing_raw = str(row.get("is_closing", "")).strip().lower()
    if closing_raw in _TRUE:
        is_closing = True
    elif closing_raw in _FALSE:
        is_closing = False
    else:
        raise ValueError(f"invalid is_closing value {row['is_closing']!r}")

    writer.add(
        game_id=game.id, book=str(row["book"]).strip(), market=market, outcome_label=label,
        decimal_odds=float(row["decimal_odds"]), line_value=line_value,
        captured_at=_parse_utc(str(row["timestamp"])), is_closing=is_closing,
    )


# --- Closing lines and the integrity gate -----------------------------------


def mark_closing_lines(
    session: Session, start: date | None = None, end: date | None = None
) -> int:
    """Flag closing snapshots; returns how many rows end up flagged.

    For every (game, market, book, outcome, line) group, the snapshot with the
    latest captured_at_utc strictly before the game's first pitch is flagged
    is_closing=True; every other snapshot in the group is reset to False
    (including stale flags from earlier runs or imports). Games without a
    known first_pitch_utc get no closing flags.

    Optional [start, end] (inclusive, on the games' official dates) restricts
    the sweep: only snapshots of games in range are cleared and re-flagged;
    flags outside the range are left untouched. Default is everything — the
    daily loop passes its two-day window so the sweep stays O(recent) as the
    snapshot table grows.
    """
    query = select(OddsSnapshot, Game.first_pitch_utc).join(
        Game, OddsSnapshot.game_id == Game.id
    )
    if start is not None:
        query = query.where(Game.official_date >= start)
    if end is not None:
        query = query.where(Game.official_date <= end)
    rows = session.execute(query).all()
    best: dict[tuple, OddsSnapshot] = {}
    for snapshot, first_pitch in rows:
        snapshot.is_closing = False
        if first_pitch is None or snapshot.captured_at_utc >= first_pitch:
            continue
        key = (
            snapshot.game_id, snapshot.market, snapshot.book,
            snapshot.outcome_label, snapshot.line_value,
        )
        current = best.get(key)
        if current is None or snapshot.captured_at_utc > current.captured_at_utc:
            best[key] = snapshot
    for snapshot in best.values():
        snapshot.is_closing = True
    return len(best)


def get_usable_odds(session: Session, game_id: int, as_of: datetime) -> list[OddsSnapshot]:
    """The ONLY sanctioned way for model/backtester code to read odds.

    Returns snapshots for the game captured strictly before BOTH the game's
    first pitch and `as_of` (the moment the caller is pretending it is).
    A game with unknown first_pitch_utc yields nothing — if we cannot prove a
    snapshot is pre-game, it does not exist. No-lookahead is enforced here
    structurally; downstream code must not query OddsSnapshot directly.
    """
    game = session.get(Game, game_id)
    if game is None:
        raise ValueError(f"no game with id {game_id}")
    if game.first_pitch_utc is None:
        return []
    return list(
        session.scalars(
            select(OddsSnapshot)
            .where(
                OddsSnapshot.game_id == game_id,
                OddsSnapshot.captured_at_utc < game.first_pitch_utc,
                OddsSnapshot.captured_at_utc < as_of,
            )
            .order_by(OddsSnapshot.captured_at_utc)
        )
    )


def require_api_key() -> str:
    """ODDS_API_KEY from the environment, failing loudly when absent."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY environment variable is not set")
    return api_key
