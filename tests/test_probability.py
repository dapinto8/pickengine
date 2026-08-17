"""Tests for the combined probability layer."""

from datetime import date, datetime

import pytest
from sqlalchemy import Engine

from pickengine.db import create_schema, get_engine, session_scope
from pickengine.engine.elo import EloRatings
from pickengine.engine.probability import (
    blend,
    market_home_probability,
    model_home_probability,
    predict_game,
)
from pickengine.models import (
    Game,
    GameStatus,
    GameType,
    Market,
    OddsSnapshot,
    PitcherStatsSnapshot,
    Player,
    Team,
)

FIRST_PITCH = datetime(2024, 6, 15, 23, 10)
P_HOME_24 = 1 / (1 + 10 ** (-24 / 400))  # equal teams, home advantage only
P_HOME_48 = 1 / (1 + 10 ** (-48 / 400))  # home advantage + 24 Elo pitching edge


def h2h(book: str, label: str, odds: float, at: datetime) -> OddsSnapshot:
    return OddsSnapshot(
        game_id=1, book=book, market=Market.H2H, outcome_label=label,
        decimal_odds=odds, line_value=None, captured_at_utc=at, is_closing=False,
    )


def test_blend_weights() -> None:
    assert blend(0.6, 0.5) == pytest.approx(0.53)  # 0.3 model + 0.7 market
    assert blend(0.6, None) == 0.6


def test_model_home_probability() -> None:
    assert model_home_probability(1500, 1500, 0.0) == pytest.approx(P_HOME_24)
    assert model_home_probability(1500, 1500, 24.0) == pytest.approx(P_HOME_48)


def test_market_probability_prefers_sharp_book() -> None:
    t = datetime(2024, 6, 15, 20, 0)
    snaps = [
        h2h("pinnacle", "New York Mets", 1.91, t),
        h2h("pinnacle", "Atlanta Braves", 1.91, t),
        h2h("draftkings", "New York Mets", 2.30, t),
        h2h("draftkings", "Atlanta Braves", 1.60, t),
    ]
    result = market_home_probability(snaps, "New York Mets", "Atlanta Braves")
    assert result is not None
    p, source = result
    assert source == "pinnacle"
    assert p == pytest.approx(0.5)


def test_market_probability_uses_latest_quote_per_book() -> None:
    early, late = datetime(2024, 6, 15, 12, 0), datetime(2024, 6, 15, 22, 0)
    snaps = [
        h2h("pinnacle", "New York Mets", 2.50, early),  # stale, superseded
        h2h("pinnacle", "New York Mets", 1.91, late),
        h2h("pinnacle", "Atlanta Braves", 1.91, late),
    ]
    p, _ = market_home_probability(snaps, "New York Mets", "Atlanta Braves")
    assert p == pytest.approx(0.5)


def test_market_probability_median_fallback_and_none() -> None:
    t = datetime(2024, 6, 15, 20, 0)
    snaps = [
        h2h("draftkings", "New York Mets", 1.91, t),
        h2h("draftkings", "Atlanta Braves", 1.91, t),
        h2h("betmgm", "New York Mets", 2.10, t),
        h2h("betmgm", "Atlanta Braves", 1.75, t),
        h2h("caesars", "New York Mets", 2.40, t),  # one-sided: excluded
    ]
    p, source = market_home_probability(snaps, "New York Mets", "Atlanta Braves")
    assert source == "median(2 books)"
    mgm_home = (1 / 2.10) / (1 / 2.10 + 1 / 1.75)
    assert p == pytest.approx((0.5 + mgm_home) / 2)

    assert market_home_probability([], "New York Mets", "Atlanta Braves") is None


@pytest.fixture
def engine() -> Engine:
    engine = get_engine(":memory:")
    create_schema(engine)
    with session_scope(engine) as session:
        mets = Team(mlb_id=121, abbreviation="NYM", name="New York Mets", league="NL")
        braves = Team(mlb_id=144, abbreviation="ATL", name="Atlanta Braves", league="NL")
        starter = Player(mlb_id=656849, name="David Peterson", position="P")
        session.add_all([mets, braves, starter])
        session.flush()
        session.add(
            Game(
                id=1, mlb_game_pk=745900, date_utc=date(2024, 6, 15), season=2024,
                game_type=GameType.REGULAR, home_team_id=mets.id, away_team_id=braves.id,
                home_starter_player_id=starter.id, away_starter_player_id=None,
                status=GameStatus.SCHEDULED, first_pitch_utc=FIRST_PITCH,
            )
        )
        # fip 3.0 at ip 40 -> shrunk 3.6 -> +24 Elo vs the unknown away starter.
        session.add(
            PitcherStatsSnapshot(
                player_id=starter.id, as_of_date=date(2024, 6, 15), season=2024,
                ip=40.0, fip=3.0, k_per_9=9.0, bb_per_9=3.0, games_started=7,
            )
        )
        # A misleading later snapshot that must never be picked for this game.
        session.add(
            PitcherStatsSnapshot(
                player_id=starter.id, as_of_date=date(2024, 6, 16), season=2024,
                ip=47.0, fip=9.99, k_per_9=9.0, bb_per_9=3.0, games_started=8,
            )
        )
        session.add(h2h("pinnacle", "New York Mets", 1.91, datetime(2024, 6, 15, 20, 0)))
        session.add(h2h("pinnacle", "Atlanta Braves", 1.91, datetime(2024, 6, 15, 20, 0)))
        # Post-first-pitch quote: must be invisible to predictions.
        session.add(h2h("pinnacle", "New York Mets", 1.20, datetime(2024, 6, 15, 23, 30)))
    return engine


def test_predict_game_end_to_end(engine: Engine) -> None:
    as_of = datetime(2024, 6, 15, 21, 0)
    with session_scope(engine) as session:
        pred = predict_game(session, EloRatings([]), game_id=1, as_of=as_of)

    assert pred.home_rating == 1500
    assert pred.away_rating == 1500
    assert pred.pitching_adjustment == pytest.approx(24.0)
    assert pred.p_model == pytest.approx(P_HOME_48)
    assert pred.p_market == pytest.approx(0.5)  # post-pitch 1.20 quote ignored
    assert pred.market_source == "pinnacle"
    assert pred.p_blend == pytest.approx(0.3 * P_HOME_48 + 0.7 * 0.5)


def test_predict_game_snapshot_cutoff_before_game_day(engine: Engine) -> None:
    # Predicting the day before: the 06-15 snapshot summarizes games through
    # 06-14, which may still be unplayed at prediction time -> not usable.
    as_of = datetime(2024, 6, 14, 12, 0)
    with session_scope(engine) as session:
        pred = predict_game(session, EloRatings([]), game_id=1, as_of=as_of)
    assert pred.pitching_adjustment == 0.0  # falls back to league average
    assert pred.p_market is None  # odds only captured later
    assert pred.p_blend == pred.p_model


def test_predict_game_unknown_game_raises(engine: Engine) -> None:
    with session_scope(engine) as session, pytest.raises(ValueError, match="no game"):
        predict_game(session, EloRatings([]), game_id=999, as_of=datetime(2024, 6, 15))
