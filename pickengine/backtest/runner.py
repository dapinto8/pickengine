"""Backtest runner: chronological replay of the full pipeline.

Walks games in first-pitch order over a date range. For each game it:
1. Builds the pre-game state (ratings, pitcher info, odds snapshots) using
   only data timestamped before that game's first pitch — the no-lookahead
   rule is enforced here structurally, not by convention.
2. Runs devig -> elo/pitching -> probability -> selection.
3. Records simulated picks, then updates ratings with the game result.

Produces a pick ledger that evaluation.py turns into CLV, Brier, ROI, and
drawdown numbers. Fully deterministic for a given database and config.
"""
