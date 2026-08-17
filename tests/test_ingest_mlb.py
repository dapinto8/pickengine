"""Tests for MLB StatsAPI ingestion, using saved JSON fixtures — no network."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, select

from pickengine.db import create_schema, get_engine, session_scope
from pickengine.ingest.mlb import (
    StatsApiClient,
    compute_fip,
    innings_to_outs,
    sync_pitcher_snapshots,
    sync_schedule,
    sync_teams,
)
from pickengine.models import Game, GameStatus, PitcherStatsSnapshot, Player, Team

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClient:
    """Stands in for StatsApiClient: serves fixtures keyed by path prefix."""

    def __init__(self, routes: dict[str, str]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((path, params))
        for prefix, fixture in self.routes.items():
            if path.startswith(prefix):
                return json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
        raise AssertionError(f"unexpected API path in test: {path}")


@pytest.fixture
def engine() -> Engine:
    engine = get_engine(":memory:")
    create_schema(engine)
    return engine


def test_innings_to_outs() -> None:
    assert innings_to_outs("6.0") == 18
    assert innings_to_outs("5.1") == 16
    assert innings_to_outs("0.2") == 2
    assert innings_to_outs("7") == 21
    with pytest.raises(ValueError):
        innings_to_outs("5.4")


def test_compute_fip_known_value() -> None:
    # (13*3 + 3*(5+1) - 2*11) / (34/3) + 3.24 (2024 constant)
    assert compute_fip(hr=3, bb=5, hbp=1, so=11, outs=34, season=2024) == pytest.approx(
        105 / 34 + 3.24
    )
    # Unknown season falls back to the default constant.
    assert compute_fip(hr=0, bb=0, hbp=0, so=0, outs=27, season=1999) == pytest.approx(3.15)


def test_sync_teams_upserts_and_is_idempotent(engine: Engine) -> None:
    client = FakeClient({"teams": "teams.json"})
    with session_scope(engine) as session:
        assert sync_teams(session, client) == 3
        assert sync_teams(session, client) == 3
    with session_scope(engine) as session:
        teams = {t.mlb_id: t for t in session.scalars(select(Team))}
        assert len(teams) == 3
        assert teams[147].league == "AL"
        assert teams[121].abbreviation == "NYM"


def test_sync_schedule(engine: Engine) -> None:
    client = FakeClient({"teams": "teams.json", "schedule": "schedule.json"})
    with session_scope(engine) as session:
        sync_teams(session, client)
        counts = sync_schedule(session, client, date(2024, 6, 15), date(2024, 6, 15))

    assert counts == {
        "games": 3,
        "postponed": 1,
        "missing_probables": 1,  # scheduled game's home side; postponed game not counted
        "skipped_game_type": 1,  # the spring-training game
    }

    with session_scope(engine) as session:
        games = {g.mlb_game_pk: g for g in session.scalars(select(Game))}
        assert set(games) == {745001, 745002, 745003}

        final = games[745001]
        assert final.status is GameStatus.FINAL
        assert (final.home_score, final.away_score) == (3, 5)
        assert final.first_pitch_utc == datetime(2024, 6, 15, 17, 10)
        assert final.date_utc == date(2024, 6, 15)

        scheduled = games[745002]
        assert scheduled.status is GameStatus.SCHEDULED
        assert scheduled.home_score is None
        assert scheduled.home_starter_player_id is None
        assert scheduled.away_starter_player_id is not None

        assert games[745003].status is GameStatus.POSTPONED

        players = {p.mlb_id for p in session.scalars(select(Player))}
        assert players == {665742, 656849, 543037}

    # Re-sync is idempotent: same rows, no duplicates.
    with session_scope(engine) as session:
        sync_schedule(session, client, date(2024, 6, 15), date(2024, 6, 15))
    with session_scope(engine) as session:
        assert len(session.scalars(select(Game)).all()) == 3


def _seed_pitcher_and_game(engine: Engine, game_date: date) -> None:
    with session_scope(engine) as session:
        home = Team(mlb_id=121, abbreviation="NYM", name="New York Mets", league="NL")
        away = Team(mlb_id=144, abbreviation="ATL", name="Atlanta Braves", league="NL")
        pitcher = Player(mlb_id=656849, name="David Peterson", position="P")
        session.add_all([home, away, pitcher])
        session.flush()
        session.add(
            Game(
                mlb_game_pk=745900,
                date_utc=game_date,
                season=game_date.year,
                game_type="regular",
                home_team_id=home.id,
                away_team_id=away.id,
                home_starter_player_id=pitcher.id,
                status=GameStatus.SCHEDULED,
            )
        )


def test_sync_pitcher_snapshots_excludes_as_of_date(engine: Engine) -> None:
    _seed_pitcher_and_game(engine, date(2024, 6, 15))
    client = FakeClient({"people/656849/stats": "gamelog.json"})

    # as_of 2024-06-14: only the 06-01 and 06-07 starts are known.
    with session_scope(engine) as session:
        counts = sync_pitcher_snapshots(session, client, date(2024, 6, 14))
    assert counts == {"snapshots": 1, "skipped_no_data": 0}

    with session_scope(engine) as session:
        snap = session.scalars(select(PitcherStatsSnapshot)).one()
        assert snap.as_of_date == date(2024, 6, 14)
        assert snap.ip == pytest.approx(34 / 3)
        assert snap.games_started == 2
        assert snap.last_start_date == date(2024, 6, 7)
        assert snap.k_per_9 == pytest.approx(11 * 27 / 34)
        assert snap.bb_per_9 == pytest.approx(5 * 27 / 34)
        assert snap.fip == pytest.approx(105 / 34 + 3.24)
        assert snap.xfip_proxy is None


def test_sync_pitcher_snapshots_full_and_upsert(engine: Engine) -> None:
    _seed_pitcher_and_game(engine, date(2024, 6, 16))
    client = FakeClient({"people/656849/stats": "gamelog.json"})

    with session_scope(engine) as session:
        sync_pitcher_snapshots(session, client, date(2024, 6, 15))
        sync_pitcher_snapshots(session, client, date(2024, 6, 15))  # upsert, no dupes

    with session_scope(engine) as session:
        snap = session.scalars(select(PitcherStatsSnapshot)).one()
        assert snap.ip == pytest.approx(55 / 3)  # all three starts: 18+16+21 outs
        assert snap.games_started == 3
        assert snap.last_start_date == date(2024, 6, 14)


def test_sync_pitcher_snapshots_skips_pitcher_with_no_prior_games(engine: Engine) -> None:
    _seed_pitcher_and_game(engine, date(2024, 4, 1))
    client = FakeClient({"people/656849/stats": "gamelog.json"})
    with session_scope(engine) as session:
        counts = sync_pitcher_snapshots(session, client, date(2024, 4, 1))
    assert counts == {"snapshots": 0, "skipped_no_data": 1}


def test_client_caches_responses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_get(url: str, timeout: float) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    client = StatsApiClient(cache_dir=tmp_path, min_interval_s=0)
    assert client.get("teams", {"sportId": 1}) == {"ok": True}
    assert client.get("teams", {"sportId": 1}) == {"ok": True}
    assert calls["n"] == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_client_retries_on_5xx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [500, 503, 200]

    def fake_get(url: str, timeout: float) -> httpx.Response:
        status = responses.pop(0)
        return httpx.Response(
            status, json={"ok": status == 200}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("pickengine.ingest.mlb.time.sleep", lambda _s: None)
    client = StatsApiClient(cache_dir=tmp_path, min_interval_s=0)
    assert client.get("teams") == {"ok": True}
    assert responses == []
