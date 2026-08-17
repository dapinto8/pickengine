"""pickengine: MLB-first sports betting prediction engine and backtester.

Three layers:
1. Market baseline — de-vigged consensus probabilities from bookmaker odds.
2. Probability model — Elo-style team ratings plus starting pitcher adjustments.
3. Pick selection — EV screening against best available odds with hard publishing rules.
"""

__version__ = "0.1.0"
