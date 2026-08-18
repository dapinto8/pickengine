"""Tests for the combined probability layer."""

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import Engine

from pickengine.db import create_schema, get_engine, session_scope
from pickengine.engine.devig import remove_vig_multiplicative, remove_vig_power
from pickengine.engine.elo import EloRatings
from pickengine.engine.probability import (
    blend,
    latest_h2h_quotes,
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


def test_market_probability_devig_method_power() -> None:
    """method="power" applies the power devig; on an asymmetric vigged pair
    it differs measurably from the multiplicative result."""
    t = datetime(2024, 6, 15, 20, 0)
    snaps = [
        h2h("pinnacle", "New York Mets", 1.40, t),
        h2h("pinnacle", "Atlanta Braves", 3.10, t),
    ]
    p_power, source = market_home_probability(
        snaps, "New York Mets", "Atlanta Braves", method="power"
    )
    assert source == "pinnacle"
    assert p_power == pytest.approx(remove_vig_power([1.40, 3.10])[0])
    p_mult, _ = market_home_probability(snaps, "New York Mets", "Atlanta Braves")
    assert p_mult == pytest.approx(remove_vig_multiplicative([1.40, 3.10])[0])
    # Power shrinks the longshot more, so the favorite keeps a higher p.
    assert p_power > p_mult

    with pytest.raises(ValueError, match="unknown devig method"):
        market_home_probability(snaps, "New York Mets", "Atlanta Braves", method="nope")


def test_market_probability_uses_later_complete_capture_when_line_moved() -> None:
    """Two complete captures from one book with a moved line: the later
    capture's pair is the one de-vigged."""
    t1, t2 = datetime(2024, 6, 15, 12, 0), datetime(2024, 6, 15, 22, 0)
    snaps = [
        h2h("pinnacle", "New York Mets", 2.20, t1),
        h2h("pinnacle", "Atlanta Braves", 1.70, t1),
        h2h("pinnacle", "New York Mets", 1.91, t2),
        h2h("pinnacle", "Atlanta Braves", 1.91, t2),
    ]
    p, source = market_home_probability(snaps, "New York Mets", "Atlanta Braves")
    assert source == "pinnacle"
    assert p == pytest.approx(0.5)


def test_market_probability_never_mixes_capture_times() -> None:
    """A later capture with only one side must NOT be paired with the other
    side from an earlier capture — those prices never coexisted. The complete
    earlier pair is used instead."""
    t1, t2 = datetime(2024, 6, 15, 12, 0), datetime(2024, 6, 15, 22, 0)
    snaps = [
        h2h("pinnacle", "New York Mets", 2.50, t1),
        h2h("pinnacle", "Atlanta Braves", 1.55, t1),
        h2h("pinnacle", "New York Mets", 1.91, t2),  # moved, away side not captured
    ]
    p, _ = market_home_probability(snaps, "New York Mets", "Atlanta Braves")
    # The t1 pair, not (1.91 @ t2, 1.55 @ t1) which would be ~0.44.
    assert p == pytest.approx(remove_vig_multiplicative([2.50, 1.55])[0])


def test_capture_pairing_tolerates_timestamp_jitter() -> None:
    """Archives often scrape a book's two sides seconds apart; quotes within
    H2H_PAIR_TOLERANCE pair as one capture instead of voiding the book (and
    with it the whole market)."""
    t = datetime(2024, 6, 15, 20, 0)
    snaps = [
        h2h("pinnacle", "New York Mets", 1.91, t),
        h2h("pinnacle", "Atlanta Braves", 1.91, t + timedelta(seconds=2)),
    ]
    p, source = market_home_probability(snaps, "New York Mets", "Atlanta Braves")
    assert source == "pinnacle"
    assert p == pytest.approx(0.5)


def test_one_sided_book_contributes_nothing() -> None:
    """A book that never quotes both sides in one capture yields no market."""
    snaps = [
        h2h("betmgm", "New York Mets", 2.10, datetime(2024, 6, 15, 12, 0)),
        h2h("betmgm", "New York Mets", 2.05, datetime(2024, 6, 15, 22, 0)),
    ]
    assert market_home_probability(snaps, "New York Mets", "Atlanta Braves") is None


def test_latest_h2h_quotes_pairs_share_capture_time() -> None:
    """Every book's contributed quotes come from a single captured_at_utc,
    even when the per-outcome latest quotes span captures."""
    t1, t2, t3 = (datetime(2024, 6, 15, h, 0) for h in (12, 18, 22))
    snaps = [
        h2h("pinnacle", "New York Mets", 2.20, t1),
        h2h("pinnacle", "Atlanta Braves", 1.70, t1),
        h2h("pinnacle", "Atlanta Braves", 1.80, t2),  # away-only capture
        h2h("betmgm", "New York Mets", 2.10, t2),
        h2h("betmgm", "Atlanta Braves", 1.75, t2),
        h2h("betmgm", "New York Mets", 2.15, t3),  # home-only capture
    ]
    quotes = latest_h2h_quotes(snaps)
    by_book: dict[str, set[datetime]] = {}
    for (book, _), snap in quotes.items():
        by_book.setdefault(book, set()).add(snap.captured_at_utc)
    assert by_book == {"pinnacle": {t1}, "betmgm": {t2}}  # one capture each
    assert len(quotes) == 4  # both sides for both books


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
                id=1, mlb_game_pk=745900, official_date=date(2024, 6, 15), season=2024,
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
