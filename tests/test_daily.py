"""Tests for the daily-flow pieces: no-cache client, phase report, export."""

from datetime import date
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine

from pickengine.backtest.evaluation import evaluate_phase
from pickengine.backtest.runner import run_backtest
from pickengine.db import create_schema, get_engine, session_scope
from pickengine.engine.selection import settle_picks
from pickengine.export import export_track_record_md
from pickengine.ingest.mlb import StatsApiClient
from pickengine.ingest.odds import mark_closing_lines
from pickengine.models import Phase, Team
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
