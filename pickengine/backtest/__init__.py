"""Backtesting: chronological replay and evaluation of the pick pipeline.

The backtester replays history strictly in time order, feeding the engine
only information available before each game's first pitch. Any randomness
uses explicit, deterministic seeds.
"""
