"""Backtest runner + evaluation tests."""

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from pickengine.backtest.evaluation import (
    brier_score,
    calibration_table,
    evaluate_run,
    max_drawdown,
    render_report,
)
from pickengine.backtest.runner import (
    _assert_pick_integrity,
    clear_backtest_run,
    run_backtest,
)
from pickengine.db import create_schema, get_engine, session_scope
from pickengine.engine.elo import EloRatings
from pickengine.models import (
    Game,
    GameStatus,
    GameType,
    Market,
    OddsSnapshot,
    Phase,
    Pick,
    PickStatus,
    PitcherStatsSnapshot,
    Player,
    Team,
)

HOME = "New York Mets"
AWAY = "Atlanta Braves"


def test_max_drawdown() -> None:
    assert max_drawdown([2, -1, -1, 3]) == pytest.approx(2.0)  # cum 2,1,0,3
    assert max_drawdown([1, 1, 1]) == 0.0
    assert max_drawdown([-1, -1]) == pytest.approx(2.0)
    assert max_drawdown([]) == 0.0


def test_brier_score() -> None:
    assert brier_score([(1.0, 1), (0.0, 0)]) == 0.0
    assert brier_score([(0.5, 1)]) == pytest.approx(0.25)
    assert brier_score([(0.7, 0)]) == pytest.approx(0.49)
    with pytest.raises(ValueError):
        brier_score([])


def test_calibration_table() -> None:
    pairs = [(0.55, 1), (0.58, 0), (0.95, 1), (1.0, 1)]
    table = calibration_table(pairs)
    assert [row["bucket"] for row in table] == ["50-60%", "90-100%"]
    assert table[0]["n"] == 2
    assert table[0]["actual_rate"] == pytest.approx(0.5)
    assert table[1]["n"] == 2  # p=1.0 clamps into the top bucket
    assert table[1]["actual_rate"] == 1.0


def first_pitch(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 23, 10)


def seed_day(
    session: Session, pk: int, day: date, home_score: int, away_score: int
) -> Game:
    """One final game with a favorable home price and closing lines."""
    game = Game(
        mlb_game_pk=pk, official_date=day, season=day.year, game_type=GameType.REGULAR,
        home_team_id=1, away_team_id=2, status=GameStatus.FINAL,
        home_score=home_score, away_score=away_score, first_pitch_utc=first_pitch(day),
    )
    session.add(game)
    session.flush()
    fp = first_pitch(day)

    def quote(book: str, label: str, odds: float, at: datetime, closing: bool = False) -> None:
        session.add(
            OddsSnapshot(
                game_id=game.id, book=book, market=Market.H2H, outcome_label=label,
                decimal_odds=odds, line_value=None, captured_at_utc=at, is_closing=closing,
            )
        )

    early = fp - timedelta(hours=5)  # before the 4h decision time
    late = fp - timedelta(hours=2)  # after decision time, before pitch
    quote("pinnacle", HOME, 1.91, early)
    quote("pinnacle", AWAY, 1.91, early)
    quote("betmgm", HOME, 2.10, early)
    quote("betmgm", AWAY, 1.80, early)  # books contribute complete pairs only
    # A juicier price that appears only AFTER decision time: must not be taken.
    quote("betmgm", HOME, 2.50, late)
    quote("betmgm", AWAY, 1.55, late)
    # Closing lines.
    quote("betmgm", HOME, 2.00, fp - timedelta(minutes=10), closing=True)
    quote("pinnacle", HOME, 1.90, fp - timedelta(minutes=10), closing=True)
    return game


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


def test_run_backtest_end_to_end(engine: Engine) -> None:
    d1, d2 = date(2024, 6, 15), date(2024, 6, 16)
    with session_scope(engine) as session:
        seed_day(session, 1, d1, home_score=5, away_score=3)  # home pick wins
        seed_day(session, 2, d2, home_score=2, away_score=5)  # home pick loses
        summary = run_backtest(session, d1, d2, run_id="testrun1")

    assert summary["picks"] == 2
    assert summary["won"] == 1
    assert summary["lost"] == 1

    with session_scope(engine) as session:
        picks = session.scalars(select(Pick).order_by(Pick.created_at_utc)).all()
        assert all(p.phase is Phase.BACKTEST and p.run_id == "testrun1" for p in picks)
        # Decision-time semantics: the 2.50 post-decision quote was untouchable.
        assert all(p.decimal_odds_at_pick == 2.10 for p in picks)
        assert all(p.created_at_utc == first_pitch(p.created_at_utc.date()) - timedelta(hours=4)
                   for p in picks)
        assert picks[0].status is PickStatus.WON
        assert picks[1].status is PickStatus.LOST
        # CLV vs own-book close 2.00: 2.10/2.00 - 1 = +5%.
        assert all(p.clv_pct == pytest.approx(5.0) for p in picks)


def test_multiple_runs_do_not_collide_and_clear_works(engine: Engine) -> None:
    d1 = date(2024, 6, 15)
    with session_scope(engine) as session:
        seed_day(session, 1, d1, 5, 3)
        run_backtest(session, d1, d1, run_id="run_a")
        run_backtest(session, d1, d1, run_id="run_b")
    with session_scope(engine) as session:
        by_run = {p.run_id for p in session.scalars(select(Pick))}
        assert by_run == {"run_a", "run_b"}
        assert clear_backtest_run(session, "run_a") == 1
    with session_scope(engine) as session:
        assert {p.run_id for p in session.scalars(select(Pick))} == {"run_b"}


def test_integrity_guard_catches_leaked_decision_time(engine: Engine) -> None:
    d1 = date(2024, 6, 15)
    with session_scope(engine) as session:
        game = seed_day(session, 1, d1, 5, 3)
        bad_pick = Pick(
            game_id=game.id, market=Market.H2H, outcome_label=HOME, line_value=None,
            decimal_odds_at_pick=2.10, book="betmgm", model_probability=0.53,
            market_consensus_probability=0.5, ev=0.07, stake_units=1.0,
            created_at_utc=first_pitch(d1) + timedelta(hours=1),  # AFTER first pitch
            status=PickStatus.PENDING, phase=Phase.BACKTEST, run_id="bad",
        )
        session.add(bad_pick)
        session.flush()
        with pytest.raises(RuntimeError, match="lookahead violation"):
            _assert_pick_integrity(session, EloRatings([]), bad_pick)


def test_integrity_guard_catches_unbacked_price(engine: Engine) -> None:
    d1 = date(2024, 6, 15)
    with session_scope(engine) as session:
        game = seed_day(session, 1, d1, 5, 3)
        phantom = Pick(
            game_id=game.id, market=Market.H2H, outcome_label=HOME, line_value=None,
            decimal_odds_at_pick=3.33, book="betmgm",  # no such pre-decision quote
            model_probability=0.53, market_consensus_probability=0.5, ev=0.07,
            stake_units=1.0, created_at_utc=first_pitch(d1) - timedelta(hours=4),
            status=PickStatus.PENDING, phase=Phase.BACKTEST, run_id="bad",
        )
        session.add(phantom)
        session.flush()
        with pytest.raises(RuntimeError, match="no snapshot"):
            _assert_pick_integrity(session, EloRatings([]), phantom)


def test_integrity_guard_catches_broken_starter_snapshot_helper(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression the old check could never detect: it re-ran the helper
    and verified the helper's own filter, so a broken starter_snapshot passed
    silently. The independent-oracle check must catch a helper that returns a
    snapshot dated after the cutoff."""
    d1 = date(2024, 6, 15)
    with session_scope(engine) as session:
        game = seed_day(session, 1, d1, 5, 3)
        pitcher = Player(mlb_id=656849, name="David Peterson", position="P")
        session.add(pitcher)
        session.flush()
        game.home_starter_player_id = pitcher.id
        # A pick that passes the decision-time and quote checks.
        pick = Pick(
            game_id=game.id, market=Market.H2H, outcome_label=HOME, line_value=None,
            decimal_odds_at_pick=2.10, book="betmgm", model_probability=0.53,
            market_consensus_probability=0.5, ev=0.07, stake_units=1.0,
            created_at_utc=first_pitch(d1) - timedelta(hours=4),
            status=PickStatus.PENDING, phase=Phase.BACKTEST, run_id="bad",
        )
        session.add(pick)
        session.flush()

        future = PitcherStatsSnapshot(
            player_id=pitcher.id, as_of_date=date(2024, 6, 20), season=2024,
            ip=10.0, fip=3.0, k_per_9=9.0, bb_per_9=3.0, games_started=2,
        )
        monkeypatch.setattr(
            "pickengine.backtest.runner.starter_snapshot", lambda *a, **k: future
        )
        with pytest.raises(RuntimeError, match="lookahead violation.*pitcher snapshot"):
            _assert_pick_integrity(session, EloRatings([]), pick)


def test_closing_capture_gap_metric_and_stale_warning(engine: Engine) -> None:
    d1 = date(2024, 6, 15)
    with session_scope(engine) as session:
        seed_day(session, 1, d1, 5, 3)  # closing snapshots 10 min before first pitch
        run_backtest(session, d1, d1, run_id="gaprun", write_meta=False)
        report = evaluate_run(session, "gaprun", d1, d1, timedelta(hours=4))
        capture = report["closing_capture"]
        assert capture["n"] == 1
        assert capture["median_gap_minutes"] == pytest.approx(10.0)
        assert capture["stale"] is False
        assert "WARNING" not in render_report(report)

    # Closing snapshots captured 3h before first pitch: stale, warning shown.
    with session_scope(engine) as session:
        for snap in session.scalars(select(OddsSnapshot).where(OddsSnapshot.is_closing)):
            snap.captured_at_utc = first_pitch(d1) - timedelta(hours=3)
    with session_scope(engine) as session:
        report = evaluate_run(session, "gaprun", d1, d1, timedelta(hours=4))
        assert report["closing_capture"]["median_gap_minutes"] == pytest.approx(180.0)
        assert report["closing_capture"]["stale"] is True
        assert "WARNING" in render_report(report)


def test_evaluate_run_report(engine: Engine) -> None:
    d1, d2 = date(2024, 6, 15), date(2024, 6, 16)
    with session_scope(engine) as session:
        seed_day(session, 1, d1, 5, 3)
        seed_day(session, 2, d2, 2, 5)
        run_backtest(session, d1, d2, run_id="evalrun")
        report = evaluate_run(session, "evalrun", d1, d2, timedelta(hours=4))

    picks = report["picks"]
    assert picks["n"] == 2
    assert (picks["wins"], picks["losses"]) == (1, 1)
    assert picks["units_staked"] == pytest.approx(2.0)
    assert picks["units_returned"] == pytest.approx(2.10)  # one win at 2.10
    assert picks["profit_units"] == pytest.approx(0.10)
    assert picks["roi_pct"] == pytest.approx(5.0)
    # Sequence: +1.10 then -1.00 -> peak 1.10, trough 0.10 -> drawdown 1.00.
    assert picks["max_drawdown_units"] == pytest.approx(1.0)

    assert report["clv"]["n"] == 2
    assert report["clv"]["avg_pct"] == pytest.approx(5.0)
    assert report["clv"]["beat_close_pct"] == 100.0

    model = report["model"]
    assert model["n_games"] == 2
    assert model["n_with_market"] == 2
    assert 0 < model["brier_score"] < 1
    assert model["calibration"]  # at least one bucket

    assert set(report["by_month"]) == {"2024-06"}
    assert report["by_side"]["favorites"]["n"] + report["by_side"]["underdogs"]["n"] == 2
