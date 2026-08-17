"""Backtest evaluation metrics, in priority order.

Pure functions computing, from a pick ledger:
1. CLV (closing line value) — primary metric: odds taken vs the de-vigged
   closing line, per pick and aggregated.
2. Calibration — Brier score of final probabilities against outcomes,
   plus reliability buckets.
3. ROI — units won / units staked.
4. Max drawdown — worst peak-to-trough decline of the cumulative units curve.

The closing line is the last odds snapshot captured before first pitch;
snapshots at or after first pitch never count as closing lines.
"""
