"""Vig removal: bookmaker odds -> fair probabilities. Pure functions.

Input is the full set of decimal odds covering ALL outcomes of one market at
one book (2 outcomes for h2h/totals/runline). Output probabilities sum to 1.

Two methods:
- multiplicative: normalize implied probabilities by their sum. Simple,
  standard, but keeps the favorite-longshot bias.
- power: find exponent k with sum((1/odds_i)^k) = 1. Shrinks longshot
  probabilities more, which better matches observed closing-line behavior.
"""

from collections.abc import Sequence


def implied_probabilities(odds: Sequence[float]) -> list[float]:
    """Raw implied probabilities 1/odds (sum > 1 when the book has vig)."""
    _validate(odds)
    return [1 / o for o in odds]


def remove_vig_multiplicative(odds: Sequence[float]) -> list[float]:
    """No-vig probabilities by proportional normalization."""
    implied = implied_probabilities(odds)
    total = sum(implied)
    return [p / total for p in implied]


def remove_vig_power(odds: Sequence[float], tol: float = 1e-12) -> list[float]:
    """No-vig probabilities via the power method.

    Solves sum((1/odds_i)^k) = 1 for k by bisection (k > 1 for a vigged
    market; k < 1 handles the rare arbitrage case where implied sum < 1).
    """
    implied = implied_probabilities(odds)

    def overround(k: float) -> float:
        return sum(p**k for p in implied) - 1

    lo, hi = 1e-3, 1e3
    if overround(lo) < 0 or overround(hi) > 0:
        raise ValueError(f"power devig cannot bracket a solution for odds {list(odds)}")
    for _ in range(200):
        mid = (lo + hi) / 2
        value = overround(mid)
        if abs(value) < tol:
            break
        if value > 0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    powered = [p**k for p in implied]
    total = sum(powered)  # residual normalization removes bisection tolerance
    return [p / total for p in powered]


def _validate(odds: Sequence[float]) -> None:
    if len(odds) < 2:
        raise ValueError(f"need odds for all outcomes of a market, got {len(odds)}")
    if any(o <= 1.0 for o in odds):
        raise ValueError(f"decimal odds must be > 1.0, got {list(odds)}")
