"""Tests for vig removal."""

import pytest

from pickengine.engine.devig import (
    implied_probabilities,
    remove_vig_multiplicative,
    remove_vig_power,
)


def test_symmetric_two_way_returns_half() -> None:
    for method in (remove_vig_multiplicative, remove_vig_power):
        probs = method([1.91, 1.91])
        assert probs == pytest.approx([0.5, 0.5])


def test_multiplicative_known_example() -> None:
    # 1.87/2.05: implied 0.534759/0.487805, sum 1.022564
    probs = remove_vig_multiplicative([1.87, 2.05])
    assert probs == pytest.approx([0.522959, 0.477041], abs=1e-6)
    assert sum(probs) == pytest.approx(1.0)


def test_power_sums_to_one_and_reduces_longshot_share() -> None:
    odds = [1.2, 5.0]  # heavy favorite market
    mult = remove_vig_multiplicative(odds)
    power = remove_vig_power(odds)
    assert sum(power) == pytest.approx(1.0)
    # Power method corrects favorite-longshot bias: more weight to the
    # favorite, less to the longshot, than proportional scaling gives.
    assert power[0] > mult[0]
    assert power[1] < mult[1]


def test_power_handles_three_way_markets() -> None:
    probs = remove_vig_power([2.50, 3.20, 3.10])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[2] > probs[1]


def test_power_handles_arbitrage_market() -> None:
    # Implied sum < 1 (cross-book arb): solution needs exponent k < 1.
    probs = remove_vig_power([2.10, 2.10])
    assert probs == pytest.approx([0.5, 0.5])
    probs = remove_vig_power([2.2, 2.05])
    assert sum(probs) == pytest.approx(1.0)


def test_implied_probabilities() -> None:
    assert implied_probabilities([2.0, 4.0]) == pytest.approx([0.5, 0.25])


@pytest.mark.parametrize("bad", [[1.91], [1.0, 2.0], [0.5, 3.0], []])
def test_invalid_odds_raise(bad: list[float]) -> None:
    for method in (remove_vig_multiplicative, remove_vig_power, implied_probabilities):
        with pytest.raises(ValueError):
            method(bad)
