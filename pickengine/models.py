"""SQLAlchemy ORM models.

Planned tables:
- Team — MLB teams (StatsAPI team id, abbreviation, league/division).
- Game — one row per game: teams, scheduled first pitch (UTC), starting
  pitchers as announced *before* first pitch, final score, status.
- Pitcher — starting pitcher identities and per-start stats.
- OddsSnapshot — a bookmaker's odds for a game market at a capture time (UTC);
  includes book, market, side, decimal odds. Closing lines are snapshots
  captured nearest to (but never after) first pitch.
- Rating — Elo-style team rating history keyed by team and effective date.
- Pick — published picks: game, market, side, model probability, odds taken,
  stake in units, and later the closing odds for CLV.

Lookahead discipline: every row that feeds a prediction must carry the UTC
time at which the information became available, so backtests can filter to
"known before first pitch" only.
"""
