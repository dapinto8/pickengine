"""Config file + tuning grid-search tests."""

from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine

from pickengine.backtest.tuning import GRID, HOLDOUT_WARNING, run_tuning, time_split
from pickengine.config import Config, load_config, save_config
from pickengine.db import create_schema, get_engine, session_scope
from pickengine.models import Team
from tests.test_backtest import HOME, seed_day


def test_config_defaults_match_code_constants() -> None:
    config = Config()
    assert config.elo_k == 4.0
    assert config.elo_per_fip == 40.0
    assert config.blend_weight_model == 0.3
    assert config.min_ev == 0.04


def test_config_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "pickengine.toml"
    original = Config(elo_k=5.0, elo_per_fip=25.0, blend_weight_model=0.2, min_ev=0.05)
    save_config(original, path)
    assert load_config(path) == original


def test_config_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "nope.toml") == Config()


def test_config_unknown_key_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "pickengine.toml"
    path.write_text("[model]\nelo_kk = 4.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="elo_kk"):
        load_config(path)


def test_time_split_is_chronological() -> None:
    tune_end, holdout_start = time_split(date(2024, 6, 1), date(2024, 6, 10))
    assert tune_end == date(2024, 6, 7)  # earliest 7 of 10 days
    assert holdout_start == date(2024, 6, 8)
    with pytest.raises(ValueError, match="too short"):
        time_split(date(2024, 6, 1), date(2024, 6, 1))


@pytest.fixture
def engine() -> Engine:
    engine = get_engine(":memory:")
    create_schema(engine)
    with session_scope(engine) as session:
        session.add_all(
            [
                Team(id=1, mlb_id=121, abbreviation="NYM", name=HOME, league="NL"),
                Team(id=2, mlb_id=144, abbreviation="ATL", name="Atlanta Braves", league="NL"),
            ]
        )
    return engine


def test_run_tuning_without_odds_refuses(engine: Engine) -> None:
    with session_scope(engine) as session, pytest.raises(RuntimeError, match="no odds"):
        run_tuning(session, date(2024, 6, 1), date(2024, 6, 10))


def test_run_tuning_end_to_end(engine: Engine, tmp_path: Path) -> None:
    start, end = date(2024, 6, 1), date(2024, 6, 10)
    with session_scope(engine) as session:
        for i in range(10):
            day = start + timedelta(days=i)
            # Alternate winners so ratings move and records are mixed.
            home, away = (5, 3) if i % 2 == 0 else (2, 5)
            seed_day(session, 100 + i, day, home, away)

    config_path = tmp_path / "pickengine.toml"
    with session_scope(engine) as session:
        report = run_tuning(session, start, end, config_path=config_path)

    assert report["grid_size"] == 4 * 3 * 4 * 4
    assert report["tune_window"] == {"start": "2024-06-01", "end": "2024-06-07"}
    assert report["holdout_window"] == {"start": "2024-06-08", "end": "2024-06-10"}
    assert report["warning"] == HOLDOUT_WARNING

    chosen = report["chosen_params"]
    for key, values in GRID.items():
        assert chosen[key] in values

    # Config written and loadable, matching the chosen params.
    assert load_config(config_path) == Config(**chosen)

    # Both windows evaluated with CLV present (seeded closings guarantee it).
    assert report["tuning_result"]["clv"]["n"] > 0
    assert report["holdout_result"]["clv"]["n"] > 0
    assert report["tuning_result"]["clv"]["avg_pct"] == pytest.approx(5.0)
    assert report["holdout_result"]["clv"]["avg_pct"] == pytest.approx(5.0)
    # The winner was chosen on the tuning window by CLV.
    assert report["top_combos"][0]["avg_clv_pct"] == pytest.approx(5.0)
