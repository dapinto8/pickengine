"""Prediction engine: pure functions only.

Classical model — Elo-style ratings plus pitcher adjustments blended against
the de-vigged market baseline, then EV-screened for pick selection. No deep
learning, no LLM-generated probabilities. No IO in this package.
"""
