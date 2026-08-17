"""Shrinkage and Elo-conversion tests for the pitching adjustment."""

from dataclasses import dataclass

import pytest

from pickengine.engine.pitching import (
    LEAGUE_AVERAGE_FIP,
    pitcher_score,
    pitching_adjustment_elo,
    shrunk_fip,
)


@dataclass
class FakeSnapshot:
    fip: float
    ip: float


def test_shrinkage_at_prior_ip_is_half() -> None:
    # ip == SHRINKAGE_IP (40) -> weight 0.5: halfway between FIP and league avg.
    assert shrunk_fip(3.0, 40.0) == pytest.approx(3.6)
    assert shrunk_fip(5.4, 40.0) == pytest.approx(4.8)


def test_shrinkage_limits() -> None:
    assert shrunk_fip(2.0, 0.0) == LEAGUE_AVERAGE_FIP  # no innings, no signal
    # Large sample: mostly face value.
    assert shrunk_fip(3.0, 360.0) == pytest.approx(4.2 + 0.9 * (3.0 - 4.2))
    with pytest.raises(ValueError):
        shrunk_fip(3.0, -1.0)


def test_pitcher_score_unknown_is_league_average() -> None:
    assert pitcher_score(None) == LEAGUE_AVERAGE_FIP
    assert pitcher_score(FakeSnapshot(fip=3.0, ip=40.0)) == pytest.approx(3.6)


def test_adjustment_is_40_elo_per_fip() -> None:
    assert pitching_adjustment_elo(3.6, 4.6) == pytest.approx(40.0)
    assert pitching_adjustment_elo(4.6, 3.6) == pytest.approx(-40.0)
    assert pitching_adjustment_elo(4.2, 4.2) == 0.0
    assert pitching_adjustment_elo(4.0, 4.5) == pytest.approx(20.0)
