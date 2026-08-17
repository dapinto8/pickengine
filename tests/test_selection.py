"""Publishing rules and settlement tests."""

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from pickengine.db import create_schema, get_engine, session_scope
from pickengine.engine.elo import EloRatings
from pickengine.engine.selection import (
    MAX_PICKS_PER_DAY,
    STAKE_UNITS,
    expected_value,
    generate_picks,
    settle_picks,
)
from pickengine.models import (
    Game,
    GameStatus,
    GameType,
    Market,
    OddsSnapshot,
    Phase,
    Pick,
    PickStatus,
    Team,
)

DAY = date(2024, 6, 15)
FIRST_PITCH = datetime(2024, 6, 15, 23, 10)
AS_OF = datetime(2024, 6, 15, 21, 0)
QUOTED_AT = datetime(2024, 6, 15, 20, 0)

P_HOME_MODEL = 1 / (1 + 10 ** (-24 / 400))  # 1500 vs 1500 + home advantage
P_HOME_BLEND = 0.3 * P_HOME_MODEL + 0.7 * 0.5  # against a 50/50 market

HOME = "New York Mets"
AWAY = "Atlanta Braves"


@pytest.fixture
def engine() -> Engine:
    engine = get_engine(":memory:")
    create_schema(engine)
    with session_scope(engine) as session:
        session.add_all(
            [
                Team(id=1, mlb_id=121, abbreviation="NYM", name=HOME, league="NL"),
                Team(id=2, mlb_id=144, abbreviation="ATL", name=AWAY, league="NL"),
            ]
        )
    return engine


def make_game(session: Session, pk: int, first_pitch: datetime = FIRST_PITCH) -> Game:
    game = Game(
        mlb_game_pk=pk, date_utc=DAY, season=2024, game_type=GameType.REGULAR,
        home_team_id=1, away_team_id=2, status=GameStatus.SCHEDULED,
        first_pitch_utc=first_pitch,
    )
    session.add(game)
    session.flush()
    return game


def add_h2h(
    session: Session, game_id: int, book: str, label: str, odds: float,
    at: datetime = QUOTED_AT, is_closing: bool = False,
) -> None:
    session.add(
        OddsSnapshot(
            game_id=game_id, book=book, market=Market.H2H, outcome_label=label,
            decimal_odds=odds, line_value=None, captured_at_utc=at, is_closing=is_closing,
        )
    )


def add_market(session: Session, game_id: int, extra_home_odds: float | None = None) -> None:
    """Fair 1.91/1.91 pinnacle market, optionally a softer home price elsewhere."""
    add_h2h(session, game_id, "pinnacle", HOME, 1.91)
    add_h2h(session, game_id, "pinnacle", AWAY, 1.91)
    if extra_home_odds is not None:
        add_h2h(session, game_id, "betmgm", HOME, extra_home_odds)


def test_basic_pick_generation_and_fields(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = make_game(session, 1)
        add_market(session, game.id, extra_home_odds=2.10)
        add_h2h(session, game.id, "betmgm", AWAY, 1.80)
        picks = generate_picks(session, EloRatings([]), DAY, Phase.BACKTEST, AS_OF)
        assert len(picks) == 1
        p = picks[0]
        assert p.outcome_label == HOME  # away EV is negative
        assert p.book == "betmgm"
        assert p.decimal_odds_at_pick == 2.10
        assert p.model_probability == pytest.approx(P_HOME_MODEL)
        assert p.market_consensus_probability == pytest.approx(0.5)
        assert p.ev == pytest.approx(P_HOME_BLEND * 2.10 - 1)
        assert p.ev > 0.04
        assert p.stake_units == STAKE_UNITS
        assert p.created_at_utc == AS_OF
        assert p.status is PickStatus.PENDING
        assert p.phase is Phase.BACKTEST


def test_ev_threshold_filters_out_thin_edges(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = make_game(session, 1)
        add_market(session, game.id)  # best home price 1.91 -> EV ~ -2.5%
        # Just below threshold: 2.03 -> EV ~ +3.6%.
        add_h2h(session, game.id, "betmgm", HOME, 2.03)
        assert generate_picks(session, EloRatings([]), DAY, Phase.BACKTEST, AS_OF) == []


def test_daily_cap_keeps_highest_ev(engine: Engine) -> None:
    with session_scope(engine) as session:
        prices = [2.10, 2.15, 2.20, 2.25, 2.30]
        for i, price in enumerate(prices):
            game = make_game(session, 100 + i)
            add_market(session, game.id, extra_home_odds=price)
        picks = generate_picks(session, EloRatings([]), DAY, Phase.BACKTEST, AS_OF)
        assert len(picks) == MAX_PICKS_PER_DAY
        # The lowest-EV qualifier (2.10) is the one cut; ordering is EV desc.
        assert [p.decimal_odds_at_pick for p in picks] == [2.30, 2.25, 2.20, 2.15]


def test_no_two_picks_on_same_game(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = make_game(session, 1)
        add_market(session, game.id)
        # Cross-book arb: both sides clear MIN_EV, home side by more.
        add_h2h(session, game.id, "betmgm", HOME, 2.30)
        add_h2h(session, game.id, "caesars", AWAY, 2.30)
        picks = generate_picks(session, EloRatings([]), DAY, Phase.BACKTEST, AS_OF)
        assert len(picks) == 1
        assert picks[0].outcome_label == HOME
        assert picks[0].ev == pytest.approx(expected_value(P_HOME_BLEND, 2.30))


def test_rerun_is_idempotent_and_started_games_skipped(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = make_game(session, 1)
        add_market(session, game.id, extra_home_odds=2.10)
        started = make_game(session, 2, first_pitch=AS_OF - timedelta(hours=1))
        add_market(session, started.id, extra_home_odds=2.10)

        first = generate_picks(session, EloRatings([]), DAY, Phase.BACKTEST, AS_OF)
        assert [p.game_id for p in first] == [game.id]  # started game excluded
        assert generate_picks(session, EloRatings([]), DAY, Phase.BACKTEST, AS_OF) == []
    with session_scope(engine) as session:
        assert len(session.scalars(select(Pick)).all()) == 1


def test_no_market_no_pick(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = make_game(session, 1)
        add_h2h(session, game.id, "betmgm", HOME, 2.50)  # one-sided: no devig anchor
        assert generate_picks(session, EloRatings([]), DAY, Phase.BACKTEST, AS_OF) == []


def add_pick(session: Session, game_id: int, label: str = HOME) -> Pick:
    pick = Pick(
        game_id=game_id, market=Market.H2H, outcome_label=label, line_value=None,
        decimal_odds_at_pick=2.10, book="betmgm", model_probability=0.52,
        market_consensus_probability=0.5, ev=0.05, stake_units=1.0,
        created_at_utc=AS_OF, status=PickStatus.PENDING, phase=Phase.BACKTEST,
    )
    session.add(pick)
    session.flush()
    return pick


def test_settlement_outcomes_and_clv(engine: Engine) -> None:
    with session_scope(engine) as session:
        won = make_game(session, 1)
        won.status, won.home_score, won.away_score = GameStatus.FINAL, 5, 3
        lost = make_game(session, 2)
        lost.status, lost.home_score, lost.away_score = GameStatus.FINAL, 2, 5
        push = make_game(session, 3)
        push.status, push.home_score, push.away_score = GameStatus.FINAL, 4, 4
        void = make_game(session, 4)
        void.status = GameStatus.POSTPONED
        pending = make_game(session, 5)

        for g in (won, lost, push, void, pending):
            add_pick(session, g.id)

        # Closing lines: won-game has same-book AND pinnacle closers (same
        # book must win); lost-game has only pinnacle; push-game has none.
        add_h2h(session, won.id, "betmgm", HOME, 2.00, at=datetime(2024, 6, 15, 23, 0),
                is_closing=True)
        add_h2h(session, won.id, "pinnacle", HOME, 1.90, at=datetime(2024, 6, 15, 23, 5),
                is_closing=True)
        add_h2h(session, lost.id, "pinnacle", HOME, 2.20, at=datetime(2024, 6, 15, 23, 5),
                is_closing=True)

        counts = settle_picks(session, DAY)
        assert counts == {
            "won": 1, "lost": 1, "push": 1, "void": 1, "still_pending": 1, "no_closing": 1,
        }

    with session_scope(engine) as session:
        by_game = {p.game_id: p for p in session.scalars(select(Pick))}
        assert by_game[1].status is PickStatus.WON
        assert by_game[1].closing_decimal_odds == 2.00  # own book preferred
        # Took 2.10, closed 2.00: beat the close -> POSITIVE CLV.
        assert by_game[1].clv_pct == pytest.approx(5.0)

        assert by_game[2].status is PickStatus.LOST
        assert by_game[2].closing_decimal_odds == 2.20
        # Took 2.10, closed 2.20: worse than close -> negative CLV.
        assert by_game[2].clv_pct == pytest.approx(2.10 / 2.20 * 100 - 100)

        assert by_game[3].status is PickStatus.PUSH
        assert by_game[3].clv_pct is None  # no closing line available

        assert by_game[4].status is PickStatus.VOID
        assert by_game[4].closing_decimal_odds is None

        assert by_game[5].status is PickStatus.PENDING


def test_settlement_away_winner(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = make_game(session, 1)
        game.status, game.home_score, game.away_score = GameStatus.FINAL, 2, 5
        add_pick(session, game.id, label=AWAY)
        counts = settle_picks(session, DAY)
        assert counts["won"] == 1
    with session_scope(engine) as session:
        assert session.scalars(select(Pick)).one().status is PickStatus.WON
