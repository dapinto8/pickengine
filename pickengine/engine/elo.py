"""Elo-style team ratings.

Pure functions for:
- Initializing team ratings (with regression toward the mean between seasons).
- Updating ratings from a game result (K-factor, home-field advantage,
  margin-of-victory handling to be decided when implemented).
- Converting a rating difference into a win probability for the team layer.

Ratings are computed strictly chronologically: the rating used for a game
reflects only games completed before that game's first pitch.
"""
