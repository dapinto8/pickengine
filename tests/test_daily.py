"""Tests for the daily-flow pieces: no-cache client, phase report, export."""

from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine

from pickengine.backtest.evaluation import evaluate_phase
from pickengine.backtest.runner import run_backtest
from pickengine.db import create_schema, get_engine, session_scope
from pickengine.engine.elo import EloRatings
from pickengine.engine.selection import generate_picks, settle_picks
from pickengine.export import export_track_record_md
from pickengine.ingest.mlb import StatsApiClient
from pickengine.ingest.odds import mark_closing_lines
from pickengine.models import (
    Game,
    GameStatus,
    GameType,
    Market,
    OddsSnapshot,
    Phase,
    Team,
)
from tests.test_backtest import AWAY, HOME, seed_day
from tests.test_selection import add_pick


def test_client_cache_disabled_never_reads_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_get(url: str, timeout: float) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"n": calls["n"]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    client = StatsApiClient(cache_dir=tmp_path, min_interval_s=0, cache_enabled=False)
    assert client.get("teams") == {"n": 1}
    assert client.get("teams") == {"n": 2}  # refetched, not served from cache
    assert list(tmp_path.glob("*.json")) == []  # and nothing written


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


def _seed_paper_history(engine: Engine) -> None:
    """Two settled paper picks (one win, one loss) plus one pending."""
    d1, d2, d3 = date(2024, 6, 15), date(2024, 6, 16), date(2024, 6, 17)
    with session_scope(engine) as session:
        won = seed_day(session, 1, d1, 5, 3)
        lost = seed_day(session, 2, d2, 2, 5)
        pending = seed_day(session, 3, d3, 0, 0)
        pending.status = pending.status.__class__.SCHEDULED
        pending.home_score = pending.away_score = None
        for game in (won, lost, pending):
            pick = add_pick(session, game.id)
            pick.phase = Phase.PAPER
        mark_closing_lines(session)
        settle_picks(session, d1, phase=Phase.PAPER)
        settle_picks(session, d2, phase=Phase.PAPER)


def test_daily_generation_covers_utc_rollover_games(engine: Engine) -> None:
    """Integration test of the `daily` pick-generation core at 14:00 UTC.

    Three games across the (today, tomorrow) official-date window `daily`
    generates for: an afternoon game today, a late west coast game tonight
    whose first pitch is past midnight UTC but whose official date is still
    today, and tomorrow's early game. All three must be considered and none
    skipped as already started (the guard compares first_pitch_utc against
    now, never dates). Pins the official-dates-lag-UTC rollover reasoning so
    a future date-handling change can't silently drop the night slate.
    """
    today, tomorrow = date(2024, 6, 15), date(2024, 6, 16)
    now = datetime(2024, 6, 15, 14, 0)  # the daily cron moment
    schedule = [
        (today, datetime(2024, 6, 15, 17, 10)),  # afternoon game today
        (today, datetime(2024, 6, 16, 2, 40)),   # west coast: past UTC midnight
        (tomorrow, datetime(2024, 6, 16, 17, 10)),  # tomorrow's early game
    ]
    game_ids = []
    with session_scope(engine) as session:
        for pk, (official, first_pitch) in enumerate(schedule, start=1):
            game = Game(
                mlb_game_pk=pk, official_date=official, season=2024,
                game_type=GameType.REGULAR, home_team_id=1, away_team_id=2,
                status=GameStatus.SCHEDULED, first_pitch_utc=first_pitch,
            )
            session.add(game)
            session.flush()
            game_ids.append(game.id)
            for book, label, odds in [
                ("pinnacle", HOME, 1.91), ("pinnacle", AWAY, 1.91),
                ("betmgm", HOME, 2.10), ("betmgm", AWAY, 1.80),
            ]:
                session.add(
                    OddsSnapshot(
                        game_id=game.id, book=book, market=Market.H2H,
                        outcome_label=label, decimal_odds=odds, line_value=None,
                        captured_at_utc=datetime(2024, 6, 15, 12, 0), is_closing=False,
                    )
                )

        picks = []
        for day in (today, tomorrow):  # exactly what `daily` iterates
            picks += generate_picks(session, EloRatings([]), day, Phase.PAPER, now)

        assert {p.game_id for p in picks} == set(game_ids)  # none skipped as started
        assert all(p.created_at_utc == now for p in picks)


def test_evaluate_phase_reports_paper_picks(engine: Engine) -> None:
    with session_scope(engine) as session:
        assert evaluate_phase(session, Phase.PAPER) is None  # nothing yet

    _seed_paper_history(engine)
    with session_scope(engine) as session:
        report = evaluate_phase(session, Phase.PAPER, config=None)
    assert report is not None
    assert report["run_id"] == "phase-paper"
    assert report["start"] == "2024-06-15"
    assert report["end"] == "2024-06-17"
    assert report["picks"]["n"] == 3
    assert (report["picks"]["wins"], report["picks"]["losses"]) == (1, 1)
    assert report["picks"]["pending"] == 1
    assert report["clv"]["n"] == 2


def test_evaluate_phase_excludes_backtest_picks(engine: Engine) -> None:
    d1 = date(2024, 6, 15)
    with session_scope(engine) as session:
        seed_day(session, 1, d1, 5, 3)
        run_backtest(session, d1, d1, run_id="bt", write_meta=False)
    with session_scope(engine) as session:
        assert evaluate_phase(session, Phase.PAPER) is None


def test_export_track_record_includes_every_pick(engine: Engine, tmp_path: Path) -> None:
    _seed_paper_history(engine)
    out = tmp_path / "track_record.md"
    with session_scope(engine) as session:
        path = export_track_record_md(session, out)
    text = path.read_text(encoding="utf-8")

    # All three picks present — including the pending one — with results.
    assert text.count(f"{AWAY} @ {HOME}") == 3
    assert "| won |" in text
    assert "| lost |" in text
    assert "| pending |" in text
    # Summary: 1-1-0, staked 2u, profit 1.10-1.00 = +0.10u.
    assert "**Record:** 1-1-0" in text
    assert "**Profit:** +0.10u" in text
    # CLV column filled for settled picks (2.10 vs close 2.00 -> +5.0).
    assert "| +5.0 |" in text
    # Losses are not hidden: the losing row carries negative units.
    assert "| -1.00 |" in text


def test_export_track_record_empty(engine: Engine, tmp_path: Path) -> None:
    out = tmp_path / "track_record.md"
    with session_scope(engine) as session:
        export_track_record_md(session, out)
    assert "no paper picks yet" in out.read_text(encoding="utf-8")
