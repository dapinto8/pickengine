"""MLB StatsAPI ingestion.

Pulls schedules, results, probable/announced starting pitchers, and team and
pitcher stats from statsapi.mlb.com (via the `statsapi` package or direct
REST calls with httpx).

Will provide functions like:
- `ingest_schedule(session, season)` — games with scheduled first pitch in UTC.
- `ingest_results(session, date_range)` — final scores for completed games.
- `ingest_probable_pitchers(session, date)` — starters as known pre-game.

Game times arrive in various zones from the API; everything is converted to
UTC before storage. Probable-pitcher data is stamped with its capture time so
backtests can enforce the no-lookahead rule.
"""
