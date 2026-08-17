# pickengine

Sports betting prediction engine and backtester, MLB-first, that will later power a paid
tips service for the Venezuelan market.

## System layers

1. **Market baseline**: de-vigged consensus probabilities from bookmaker odds.
2. **Probability model**: Elo-style team ratings plus starting pitcher adjustments
   (classical model — NOT deep learning, NOT LLM-generated probabilities).
3. **Pick selection**: EV screening against best available odds with hard publishing rules.

## Success metrics (in priority order)

1. CLV (closing line value) — primary
2. Calibration (Brier score)
3. ROI
4. Max drawdown

## Absolute rule: no lookahead bias

The model must NEVER use information not available before first pitch. This applies to
every feature, every backtest, every query. No exceptions.

## Tech decisions

- Python 3.11+, managed with **uv**
- **SQLite via SQLAlchemy** (simple, file-based, good enough; may move to Postgres later)
- Data sources:
  - **MLB StatsAPI** (free, via the `statsapi` / MLB-StatsAPI pip package or direct REST
    calls to statsapi.mlb.com) for schedules, results, pitchers, stats
  - **The Odds API** (the-odds-api.com) for odds; historical odds snapshots come from
    their paid historical endpoints, so the ingestion layer must work from both live
    pulls and imported CSV/JSON dumps
- CLI-driven via a single entrypoint (`python -m pickengine <command>`), no web UI yet
- **pytest** for tests. Every module gets tests. Deterministic seeds everywhere
  randomness exists.
- Timezone discipline: all timestamps stored in **UTC**, game times converted explicitly.
  Venezuela is UTC-4, but storage is UTC only.
- Type hints everywhere, **ruff** for linting

## Coding style

- Small modules, pure functions where possible, side effects isolated in ingestion and
  CLI layers
- No premature abstraction, no speculative config options
- Money and odds: store decimal odds as float, probabilities as float 0–1, stakes in
  "units" as float
- Never silently swallow API errors; fail loudly with context

## Layout

```
pickengine/
  cli.py            entrypoint (typer)
  db.py             engine, session, schema creation
  models.py         SQLAlchemy models
  ingest/
    mlb.py          StatsAPI ingestion
    odds.py         The Odds API ingestion + file import
  engine/
    elo.py
    pitching.py
    probability.py  combines layers into final p
    devig.py        vig removal from market odds
    selection.py    EV screen + publishing rules
  backtest/
    runner.py
    evaluation.py   CLV, Brier, ROI, drawdown
tests/
```

## Configuration

Tunable model parameters (elo_k, elo_per_fip, blend_weight_model, min_ev) live in
`pickengine.toml` at the repo root, written by `pickengine tune` and loaded by the
pipeline CLI commands (see `pickengine/config.py`). Code constants are the defaults
when the file is absent; never hand-edit code constants to tune — use the config file.
Tuning uses a strict time-based 70/30 split; the holdout result may be looked at once
per major model change.

## Commands

- Run CLI: `uv run python -m pickengine --help`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
