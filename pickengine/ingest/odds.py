"""Odds ingestion: The Odds API (the-odds-api.com) plus file import.

Two entry paths that normalize into the same OddsSnapshot rows:
- Live pulls from The Odds API (moneyline and other MLB markets, decimal
  odds, per-bookmaker) via httpx.
- Import of CSV/JSON dumps, since historical odds snapshots come from the
  paid historical endpoints and may be delivered as file exports.

Will provide functions like:
- `ingest_live_odds(session, api_key, date)` — snapshot current odds.
- `import_odds_file(session, path)` — load a CSV/JSON dump of snapshots.

Every snapshot stores its capture timestamp in UTC; a snapshot captured at or
after first pitch is never usable as a pre-game input. API errors and
malformed rows fail loudly with context — no silent skips.
"""
