"""Pick selection: EV screening and hard publishing rules.

Pure functions that take final probabilities (probability.py) and the best
available odds per side across books, and decide what gets published:

- Expected value: EV = p * decimal_odds - 1, computed against the best
  available price.
- Hard publishing rules (thresholds to be fixed when implemented): minimum
  EV, minimum odds / maximum odds bounds, per-day pick limits, stake sizing
  in units (flat or fractional-Kelly-capped).

Rules are hard gates, not suggestions: a pick that fails any rule is not
published, ever. Output is a list of pick candidates with probability, odds
taken, and stake in units.
"""
