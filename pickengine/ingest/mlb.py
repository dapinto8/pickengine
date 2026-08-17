"""MLB StatsAPI ingestion (statsapi.mlb.com REST, no API key required).

Provides:
- `sync_teams` — upsert all MLB teams.
- `sync_schedule` — upsert games in a date range, with probable starters when
  available and final scores for completed games. Postponed / suspended /
  cancelled games are stored with status POSTPONED (our schema does not
  distinguish further; makeup games reuse the same gamePk and simply
  overwrite the row when their date is synced).
- `sync_pitcher_snapshots` — build PitcherStatsSnapshot rows for relevant
  pitchers as of a date.

Season-stats limitation (IMPORTANT): the StatsAPI `stats=season` endpoint only
returns season-to-date totals as of *now* — there is no way to ask for season
stats as they stood on an arbitrary past date. Snapshots are therefore ALWAYS
computed from per-game pitching logs (`stats=gameLog`), aggregating only games
strictly before the snapshot's as_of_date. That is lookahead-safe by
construction and works identically for live use and historical backfills, so
the game-log path is the primary (and only) implementation.

HTTP behavior: 0.5s minimum spacing between real requests, retry on 5xx and
transport errors with exponential backoff, and raw JSON responses cached to
./cache/ keyed by URL hash so repeated backfills don't rehit the API. Errors
are never swallowed: 4xx/exhausted retries raise with the URL in the message.
"""

import hashlib
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from pickengine.models import Game, GameStatus, GameType, PitcherStatsSnapshot, Player, Team

API_BASE = "https://statsapi.mlb.com/api/v1"

# Approximate FanGraphs yearly FIP constants; close enough for a rating input.
FIP_CONSTANTS = {2021: 3.17, 2022: 3.11, 2023: 3.25, 2024: 3.24, 2025: 3.15}
DEFAULT_FIP_CONSTANT = 3.15

LEAGUE_BY_ID = {103: "AL", 104: "NL"}

# StatsAPI gameType codes: R regular; F/D/L/W postseason rounds. Everything
# else (spring training, exhibitions, all-star) is skipped.
GAME_TYPE_MAP = {
    "R": GameType.REGULAR,
    "F": GameType.POSTSEASON,
    "D": GameType.POSTSEASON,
    "L": GameType.POSTSEASON,
    "W": GameType.POSTSEASON,
}


class StatsApiClient:
    """Thin StatsAPI HTTP client: file cache, polite throttle, 5xx retry."""

    def __init__(
        self,
        cache_dir: str | Path = "./cache",
        min_interval_s: float = 0.5,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        cache_enabled: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.cache_enabled = cache_enabled
        self._last_request_at = 0.0

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET {API_BASE}/{path}, returning parsed JSON (cached by URL hash).

        The cache never expires, which is exactly right for historical
        backfills and exactly wrong for live use (today's schedule and
        current-season game logs change under the same URL) — live callers
        must pass cache_enabled=False, which neither reads nor writes cache.
        """
        query = urlencode(sorted((params or {}).items()))
        url = f"{API_BASE}/{path}" + (f"?{query}" if query else "")
        cache_file = self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        if self.cache_enabled and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        data = self._fetch(url)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _fetch(self, url: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = httpx.get(url, timeout=self.timeout_s)
            except httpx.TransportError as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
            if response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"server error {response.status_code} for {url}",
                    request=response.request,
                    response=response,
                )
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(
            f"StatsAPI request failed after {self.max_retries + 1} attempts: {url}"
        ) from last_error

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()


def innings_to_outs(innings_pitched: str) -> int:
    """Convert StatsAPI innings notation ("6.1" = 6 innings + 1 out) to outs."""
    whole, dot, frac = innings_pitched.partition(".")
    outs = int(whole) * 3 + (int(frac) if dot else 0)
    if dot and int(frac) not in (0, 1, 2):
        raise ValueError(f"invalid inningsPitched value: {innings_pitched!r}")
    return outs


def compute_fip(hr: int, bb: int, hbp: int, so: int, outs: int, season: int) -> float:
    """Standard FIP: (13*HR + 3*(BB+HBP) - 2*K) / IP + per-season constant."""
    if outs <= 0:
        raise ValueError("FIP undefined with zero outs recorded")
    constant = FIP_CONSTANTS.get(season, DEFAULT_FIP_CONSTANT)
    return (13 * hr + 3 * (bb + hbp) - 2 * so) / (outs / 3) + constant


def _parse_utc(timestamp: str) -> datetime:
    """StatsAPI UTC timestamp ("2024-06-15T23:10:00Z") -> naive UTC datetime."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).replace(tzinfo=None)


def _parse_status(status: dict[str, Any]) -> GameStatus:
    detailed = status.get("detailedState", "")
    if detailed.startswith(("Postponed", "Suspended", "Cancelled")):
        return GameStatus.POSTPONED
    if status.get("abstractGameState") == "Final":
        return GameStatus.FINAL
    return GameStatus.SCHEDULED


def sync_teams(session: Session, client: StatsApiClient) -> int:
    """Upsert all MLB teams. Returns the number of teams upserted."""
    data = client.get("teams", {"sportId": 1})
    existing = {t.mlb_id: t for t in session.scalars(select(Team))}
    count = 0
    for raw in data["teams"]:
        league_id = raw["league"]["id"]
        if league_id not in LEAGUE_BY_ID:
            raise ValueError(f"unknown league id {league_id} for team {raw['name']!r}")
        team = existing.get(raw["id"])
        if team is None:
            team = Team(mlb_id=raw["id"], abbreviation="", name="", league="")
            session.add(team)
        team.abbreviation = raw["abbreviation"]
        team.name = raw["name"]
        team.league = LEAGUE_BY_ID[league_id]
        count += 1
    return count


def _upsert_player(session: Session, cache: dict[int, Player], raw: dict[str, Any]) -> Player:
    player = cache.get(raw["id"])
    if player is None:
        player = Player(mlb_id=raw["id"], name=raw["fullName"], position="P")
        session.add(player)
        session.flush()
        cache[raw["id"]] = player
    return player


def sync_schedule(
    session: Session, client: StatsApiClient, start_date: date, end_date: date
) -> dict[str, int]:
    """Upsert games (and probable-starter players) in [start_date, end_date].

    Requires teams to be synced first. Scores are stored only for FINAL games.
    Returns counts: games upserted, postponed, missing probable starters,
    and non-regular/postseason games skipped.
    """
    team_id_by_mlb = {t.mlb_id: t.id for t in session.scalars(select(Team))}
    if not team_id_by_mlb:
        raise RuntimeError("no teams in database — run sync-teams first")
    games_by_pk = {g.mlb_game_pk: g for g in session.scalars(select(Game))}
    player_cache = {p.mlb_id: p for p in session.scalars(select(Player))}

    counts = {"games": 0, "postponed": 0, "missing_probables": 0, "skipped_game_type": 0}
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, days=30):
        data = client.get(
            "schedule",
            {
                "sportId": 1,
                "startDate": chunk_start.isoformat(),
                "endDate": chunk_end.isoformat(),
                "hydrate": "probablePitcher",
            },
        )
        for day in data.get("dates", []):
            for raw in day["games"]:
                game_type = GAME_TYPE_MAP.get(raw["gameType"])
                if game_type is None:
                    counts["skipped_game_type"] += 1
                    continue
                _upsert_game(session, raw, game_type, team_id_by_mlb, player_cache,
                             games_by_pk, counts)
    return counts


def _upsert_game(
    session: Session,
    raw: dict[str, Any],
    game_type: GameType,
    team_id_by_mlb: dict[int, int],
    player_cache: dict[int, Player],
    games_by_pk: dict[int, Game],
    counts: dict[str, int],
) -> None:
    status = _parse_status(raw["status"])
    first_pitch = _parse_utc(raw["gameDate"])
    home_raw, away_raw = raw["teams"]["home"], raw["teams"]["away"]

    starters: dict[str, int | None] = {}
    for side, side_raw in (("home", home_raw), ("away", away_raw)):
        probable = side_raw.get("probablePitcher")
        starters[side] = _upsert_player(session, player_cache, probable).id if probable else None
        if probable is None and status is not GameStatus.POSTPONED:
            counts["missing_probables"] += 1

    game = games_by_pk.get(raw["gamePk"])
    if game is None:
        game = Game(
            mlb_game_pk=raw["gamePk"],
            date_utc=first_pitch.date(),
            season=int(raw["season"]),
            game_type=game_type,
            home_team_id=team_id_by_mlb[home_raw["team"]["id"]],
            away_team_id=team_id_by_mlb[away_raw["team"]["id"]],
            status=status,
        )
        session.add(game)
        games_by_pk[raw["gamePk"]] = game
    game.date_utc = first_pitch.date()
    game.season = int(raw["season"])
    game.game_type = game_type
    game.home_team_id = team_id_by_mlb[home_raw["team"]["id"]]
    game.away_team_id = team_id_by_mlb[away_raw["team"]["id"]]
    game.home_starter_player_id = starters["home"]
    game.away_starter_player_id = starters["away"]
    game.status = status
    game.first_pitch_utc = first_pitch
    game.home_score = home_raw.get("score") if status is GameStatus.FINAL else None
    game.away_score = away_raw.get("score") if status is GameStatus.FINAL else None

    counts["games"] += 1
    if status is GameStatus.POSTPONED:
        counts["postponed"] += 1


def _date_chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _relevant_pitcher_ids(session: Session, as_of_date: date) -> set[int]:
    """Pitchers who are probable starters in [as_of, as_of+2] or started in
    [as_of-30, as_of)."""
    ids: set[int] = set()
    for lo, hi in (
        (as_of_date, as_of_date + timedelta(days=2)),
        (as_of_date - timedelta(days=30), as_of_date - timedelta(days=1)),
    ):
        rows = session.execute(
            select(Game.home_starter_player_id, Game.away_starter_player_id).where(
                Game.date_utc >= lo,
                Game.date_utc <= hi,
                or_(
                    Game.home_starter_player_id.is_not(None),
                    Game.away_starter_player_id.is_not(None),
                ),
            )
        )
        for home_id, away_id in rows:
            ids.update(pid for pid in (home_id, away_id) if pid is not None)
    return ids


def sync_pitcher_snapshots(
    session: Session, client: StatsApiClient, as_of_date: date
) -> dict[str, int]:
    """Build pitcher stats snapshots for as_of_date from per-game logs.

    A snapshot aggregates the pitcher's game log for season == as_of_date.year,
    including ONLY games dated strictly before as_of_date (see module
    docstring for why game logs, not season totals). `ip` is stored as true
    decimal innings (outs / 3). `xfip_proxy` stays None until we ingest
    batted-ball data. Pitchers with no appearances before as_of_date in the
    season are skipped.

    Returns counts: snapshots written and pitchers skipped for lack of data.
    """
    season = as_of_date.year
    pitcher_ids = _relevant_pitcher_ids(session, as_of_date)
    players = {
        p.id: p for p in session.scalars(select(Player).where(Player.id.in_(pitcher_ids)))
    }
    existing = {
        s.player_id: s
        for s in session.scalars(
            select(PitcherStatsSnapshot).where(
                PitcherStatsSnapshot.as_of_date == as_of_date,
                PitcherStatsSnapshot.season == season,
            )
        )
    }

    counts = {"snapshots": 0, "skipped_no_data": 0}
    for player_id in sorted(pitcher_ids):
        player = players[player_id]
        data = client.get(
            f"people/{player.mlb_id}/stats",
            {"stats": "gameLog", "group": "pitching", "season": season},
        )
        stats_blocks = data.get("stats", [])
        splits = stats_blocks[0].get("splits", []) if stats_blocks else []

        outs = so = bb = hr = hbp = games_started = 0
        last_start: date | None = None
        for split in splits:
            split_date = date.fromisoformat(split["date"])
            if split_date >= as_of_date:
                continue
            stat = split["stat"]
            outs += innings_to_outs(stat["inningsPitched"])
            so += stat["strikeOuts"]
            bb += stat["baseOnBalls"]
            hr += stat["homeRuns"]
            hbp += stat["hitByPitch"]
            if stat["gamesStarted"]:
                games_started += stat["gamesStarted"]
                last_start = max(last_start, split_date) if last_start else split_date

        if outs == 0:
            counts["skipped_no_data"] += 1
            continue

        snapshot = existing.get(player_id)
        if snapshot is None:
            snapshot = PitcherStatsSnapshot(
                player_id=player_id, as_of_date=as_of_date, season=season,
                ip=0.0, fip=0.0, k_per_9=0.0, bb_per_9=0.0, games_started=0,
            )
            session.add(snapshot)
        snapshot.ip = outs / 3
        snapshot.fip = compute_fip(hr=hr, bb=bb, hbp=hbp, so=so, outs=outs, season=season)
        snapshot.xfip_proxy = None
        snapshot.k_per_9 = so * 27 / outs
        snapshot.bb_per_9 = bb * 27 / outs
        snapshot.games_started = games_started
        snapshot.last_start_date = last_start
        counts["snapshots"] += 1
    return counts
