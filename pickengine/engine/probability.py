"""Final probability: combine Elo + pitching with the market baseline.

Model side: p_model = elo_expectation(home_rating + HOME_ADVANTAGE_ELO +
pitching_adjustment, away_rating), where the pitching adjustment comes from
the two starters' shrunk FIPs.

Market side: from odds usable at prediction time (via get_usable_odds — the
only sanctioned odds read), take each book's latest COMPLETE capture (both
h2h sides quoted within H2H_PAIR_TOLERANCE of each other — never a pair
mixed across distinct capture passes), use the first book in
SHARP_BOOK_PRIORITY with such a pair, de-vig it; if no priority book is
available, fall back to the median de-vigged home probability across all
books with a complete capture. Pricing a single side (selection layer) uses
latest_h2h_prices instead — the freshest quote per outcome.

Why blend: our standalone model is weaker than the market — the de-vigged
consensus is the best single estimate available. Blending with a modest model
weight (BLEND_WEIGHT_MODEL) keeps us anchored to the market while letting the
model push the probability where it genuinely disagrees; the weight is a
backtest tuning target. Without any market quote, p_blend falls back to
p_model alone (the selection layer decides whether that is publishable).

This module performs read-only DB access to assemble inputs (`predict_game`);
the math itself is in pure helpers.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from pickengine.engine.devig import remove_vig_multiplicative, remove_vig_power
from pickengine.engine.elo import HOME_ADVANTAGE_ELO, EloRatings, elo_expectation
from pickengine.engine.pitching import ELO_PER_FIP, pitcher_score, pitching_adjustment_elo
from pickengine.ingest.odds import get_usable_odds
from pickengine.models import Game, Market, OddsSnapshot, PitcherStatsSnapshot, Team

BLEND_WEIGHT_MODEL = 0.3
SHARP_BOOK_PRIORITY = ("pinnacle",)

# Default vig-removal method; a tuning knob via Config.devig_method.
DEVIG_METHOD = "multiplicative"
DEVIG_FUNCS = {"multiplicative": remove_vig_multiplicative, "power": remove_vig_power}


@dataclass(frozen=True)
class GamePrediction:
    game_id: int
    p_model: float
    p_market: float | None
    p_blend: float
    home_rating: float
    away_rating: float
    pitching_adjustment: float
    market_source: str | None  # book key, "median(N books)", or None


def model_home_probability(
    home_rating: float, away_rating: float, pitching_adjustment: float
) -> float:
    return elo_expectation(home_rating + HOME_ADVANTAGE_ELO + pitching_adjustment, away_rating)


def blend(
    p_model: float, p_market: float | None, model_weight: float = BLEND_WEIGHT_MODEL
) -> float:
    if p_market is None:
        return p_model
    return model_weight * p_model + (1 - model_weight) * p_market


# Quotes captured within this window count as one capture when pairing a
# book's two h2h sides for de-vigging. Live pulls stamp every outcome of a
# pull with one exact time, but imported archives often scrape the two sides
# seconds apart — exact-equality grouping would leave every book of such a
# file permanently "incomplete" and void the whole market.
H2H_PAIR_TOLERANCE = timedelta(seconds=60)


def latest_h2h_prices(snapshots: list[OddsSnapshot]) -> dict[tuple[str, str], OddsSnapshot]:
    """Each book's latest h2h quote per outcome, keyed by (book, outcome_label).

    For PRICING only (best available price, EV, decimal_odds_at_pick): a
    single side needs no coexisting pair, and anything older than the book's
    latest quote for that side is a stale price the book may have withdrawn.
    De-vigging must instead use latest_h2h_quotes (complete captures only).
    """
    latest: dict[tuple[str, str], OddsSnapshot] = {}
    for snap in sorted(snapshots, key=lambda s: s.captured_at_utc):
        if snap.market is Market.H2H:
            latest[(snap.book, snap.outcome_label)] = snap
    return latest


def latest_h2h_quotes(snapshots: list[OddsSnapshot]) -> dict[tuple[str, str], OddsSnapshot]:
    """Each book's latest COMPLETE h2h capture, keyed by (book, outcome_label).

    A book's snapshots are grouped into captures (quotes within
    H2H_PAIR_TOLERANCE of each other), and only its latest capture quoting
    BOTH sides contributes. Taking the latest quote per outcome independently
    would pair prices from different capture times whenever a line moved
    between intra-day pulls — prices that never coexisted, producing a
    phantom vig (or phantom arb) in the de-vig. A book that never quotes
    both sides in one capture contributes nothing.
    """
    by_book: dict[str, list[OddsSnapshot]] = defaultdict(list)
    for snap in snapshots:
        if snap.market is Market.H2H:
            by_book[snap.book].append(snap)

    result: dict[tuple[str, str], OddsSnapshot] = {}
    for book, snaps in by_book.items():
        snaps.sort(key=lambda s: s.captured_at_utc)
        captures: list[dict[str, OddsSnapshot]] = []
        cluster_end: datetime | None = None
        for snap in snaps:
            if cluster_end is None or snap.captured_at_utc - cluster_end > H2H_PAIR_TOLERANCE:
                captures.append({})
            captures[-1][snap.outcome_label] = snap
            cluster_end = snap.captured_at_utc
        for by_label in reversed(captures):  # latest complete capture wins
            if len(by_label) >= 2:
                result.update({(book, label): snap for label, snap in by_label.items()})
                break
    return result


def market_home_probability(
    snapshots: list[OddsSnapshot], home_name: str, away_name: str,
    method: str = DEVIG_METHOD,
) -> tuple[float, str] | None:
    """De-vigged home win probability from h2h snapshots, or None.

    Uses each book's latest complete capture (both sides within
    H2H_PAIR_TOLERANCE, via latest_h2h_quotes), so the de-vig inputs always
    coexisted. Book choice: first of SHARP_BOOK_PRIORITY with a complete
    pair, else median across all books with one. `method` picks the
    vig-removal function (see DEVIG_FUNCS).
    """
    devig = DEVIG_FUNCS.get(method)
    if devig is None:
        raise ValueError(f"unknown devig method {method!r} (expected {sorted(DEVIG_FUNCS)})")
    latest = latest_h2h_quotes(snapshots)

    by_book: dict[str, float] = {}
    for book in {book for book, _ in latest}:
        home_snap = latest.get((book, home_name))
        away_snap = latest.get((book, away_name))
        if home_snap and away_snap:
            by_book[book] = devig([home_snap.decimal_odds, away_snap.decimal_odds])[0]

    if not by_book:
        return None
    for book in SHARP_BOOK_PRIORITY:
        if book in by_book:
            return by_book[book], book
    return median(by_book.values()), f"median({len(by_book)} books)"


def starter_snapshot(
    session: Session, player_id: int | None, game_date: date, as_of: datetime
) -> PitcherStatsSnapshot | None:
    """Latest snapshot usable for this game at this time.

    Contract: returns the player's snapshot with the greatest as_of_date
    satisfying as_of_date <= min(game_date, as_of.date()), or None when the
    player is unknown or no snapshot qualifies. game_date is the game's
    official date; snapshot as_of_dates share the same official-date basis.
    Both bounds are load-bearing for no-lookahead: as_of_date <= game date
    because the snapshot knows nothing from the game day itself, and
    as_of_date <= as_of's date because predicting a game a day ahead must not
    use a snapshot that summarizes games still unplayed at prediction time.
    This is the single sanctioned way to pick a starter's stats for a
    prediction; the backtest runner re-verifies its choice against an
    independent query (see backtest.runner._assert_pick_integrity).
    """
    if player_id is None:
        return None
    cutoff = min(game_date, as_of.date())
    return session.scalars(
        select(PitcherStatsSnapshot)
        .where(
            PitcherStatsSnapshot.player_id == player_id,
            PitcherStatsSnapshot.as_of_date <= cutoff,
        )
        .order_by(PitcherStatsSnapshot.as_of_date.desc())
        .limit(1)
    ).first()


def predict_game(
    session: Session,
    elo: EloRatings,
    game_id: int,
    as_of: datetime,
    *,
    blend_weight: float = BLEND_WEIGHT_MODEL,
    elo_per_fip: float = ELO_PER_FIP,
    devig_method: str = DEVIG_METHOD,
) -> GamePrediction:
    """Assemble the full prediction for one game as of a moment in time."""
    game = session.get(Game, game_id)
    if game is None:
        raise ValueError(f"no game with id {game_id}")

    home_rating = elo.get_rating(game.home_team_id, as_of)
    away_rating = elo.get_rating(game.away_team_id, as_of)
    home_snap = starter_snapshot(session, game.home_starter_player_id, game.official_date, as_of)
    away_snap = starter_snapshot(session, game.away_starter_player_id, game.official_date, as_of)
    adjustment = pitching_adjustment_elo(
        pitcher_score(home_snap), pitcher_score(away_snap), elo_per_fip=elo_per_fip
    )
    p_model = model_home_probability(home_rating, away_rating, adjustment)

    home_name = session.get(Team, game.home_team_id).name
    away_name = session.get(Team, game.away_team_id).name
    market = market_home_probability(
        get_usable_odds(session, game_id, as_of), home_name, away_name, method=devig_method
    )
    p_market, source = market if market is not None else (None, None)

    return GamePrediction(
        game_id=game_id,
        p_model=p_model,
        p_market=p_market,
        p_blend=blend(p_model, p_market, blend_weight),
        home_rating=home_rating,
        away_rating=away_rating,
        pitching_adjustment=adjustment,
        market_source=source,
    )
