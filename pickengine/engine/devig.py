"""Vig removal: bookmaker odds -> fair consensus probabilities.

Pure functions for:
- Converting decimal odds to implied probabilities.
- Removing the bookmaker margin from a market (proportional/multiplicative
  method first; power or Shin methods can be added later if they earn it).
- Aggregating de-vigged probabilities across books into a consensus baseline.

Input: decimal odds as floats. Output: probabilities as floats in (0, 1)
that sum to 1 across the sides of a market.
"""
