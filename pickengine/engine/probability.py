"""Final probability: combine Elo + pitching with the market baseline.

Model side: p_model = elo_expectation(home_rating + HOME_ADVANTAGE_ELO +
pitching_adjustment, away_rating), where the pitching adjustment comes from
the two starters' shrunk FIPs.

Market side: from odds usable at prediction time (via get_usable_odds — the
only sanctioned odds read), take the latest h2h quote per (book, outcome),
use the first book in SHARP_BOOK_PRIORITY that quotes both sides, de-vig it;
if no priority book is available, fall back to the median de-vigged home
probability across all complete books.

Why blend: our standalone model is weaker than the market — the de-vigged
consensus is the best single estimate available. Blending with a modest model
weight (BLEND_WEIGHT_MODEL) keeps us anchored to the market while letting the
model push the probability where it genuinely disagrees; the weight is a
backtest tuning target. Without any market quote, p_blend falls back to
p_model alone (the selection layer decides whether that is publishable).

This module performs read-only DB access to assemble inputs (`predict_game`);
the math itself is in pure helpers.
"""

from dataclasses import dataclass
from datetime import datetime
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from pickengine.engine.devig import remove_vig_multiplicative
from pickengine.engine.elo import HOME_ADVANTAGE_ELO, EloRatings, elo_expectation
from pickengine.engine.pitching import ELO_PER_FIP, pitcher_score, pitching_adjustment_elo
from pickengine.ingest.odds import get_usable_odds
from pickengine.models import Game, Market, OddsSnapshot, PitcherStatsSnapshot, Team

BLEND_WEIGHT_MODEL = 0.3
SHARP_BOOK_PRIORITY = ("pinnacle",)


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


def latest_h2h_quotes(snapshots: list[OddsSnapshot]) -> dict[tuple[str, str], OddsSnapshot]:
    """Each book's latest h2h quote, keyed by (book, outcome_label)."""
    latest: dict[tuple[str, str], OddsSnapshot] = {}
    for snap in sorted(snapshots, key=lambda s: s.captured_at_utc):
        if snap.market is Market.H2H:
            latest[(snap.book, snap.outcome_label)] = snap
    return latest


def market_home_probability(
    snapshots: list[OddsSnapshot], home_name: str, away_name: str
) -> tuple[float, str] | None:
    """De-vigged home win probability from h2h snapshots, or None.

    Uses each book's latest quote per outcome. Book choice: first of
    SHARP_BOOK_PRIORITY with both sides quoted, else median across all
    complete books.
    """
    latest = latest_h2h_quotes(snapshots)

    by_book: dict[str, float] = {}
    for book in {book for book, _ in latest}:
        home_snap = latest.get((book, home_name))
        away_snap = latest.get((book, away_name))
        if home_snap and away_snap:
            by_book[book] = remove_vig_multiplicative(
                [home_snap.decimal_odds, away_snap.decimal_odds]
            )[0]

    if not by_book:
        return None
    for book in SHARP_BOOK_PRIORITY:
        if book in by_book:
            return by_book[book], book
    return median(by_book.values()), f"median({len(by_book)} books)"


def _starter_snapshot(
    session: Session, player_id: int | None, game_date: datetime, as_of: datetime
) -> PitcherStatsSnapshot | None:
    """Latest snapshot usable for this game at this time.

    as_of_date <= game date (snapshot knows nothing from the game day itself)
    AND as_of_date <= as_of's date — predicting a game a day ahead must not
    use a snapshot that summarizes games still unplayed at prediction time.
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
) -> GamePrediction:
    """Assemble the full prediction for one game as of a moment in time."""
    game = session.get(Game, game_id)
    if game is None:
        raise ValueError(f"no game with id {game_id}")

    home_rating = elo.get_rating(game.home_team_id, as_of)
    away_rating = elo.get_rating(game.away_team_id, as_of)
    home_snap = _starter_snapshot(session, game.home_starter_player_id, game.date_utc, as_of)
    away_snap = _starter_snapshot(session, game.away_starter_player_id, game.date_utc, as_of)
    adjustment = pitching_adjustment_elo(
        pitcher_score(home_snap), pitcher_score(away_snap), elo_per_fip=elo_per_fip
    )
    p_model = model_home_probability(home_rating, away_rating, adjustment)

    home_name = session.get(Team, game.home_team_id).name
    away_name = session.get(Team, game.away_team_id).name
    market = market_home_probability(
        get_usable_odds(session, game_id, as_of), home_name, away_name
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
