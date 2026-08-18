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
  Venezuela is UTC-4, but storage is UTC only. One deliberate exception:
  `Game.official_date` (and the pitcher snapshot dates on the same basis) is MLB's
  official LOCAL calendar date, used for identity/grouping only — never derive it from
  a UTC timestamp, and use only `first_pitch_utc` for time-ordering and lookahead
  checks (see `models.py`).
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

Tunable model parameters (elo_k, elo_per_fip, blend_weight_model, min_ev,
devig_method) live in
`pickengine.toml` at the repo root, written by `pickengine tune` and loaded by the
pipeline CLI commands (see `pickengine/config.py`). Code constants are the defaults
when the file is absent; never hand-edit code constants to tune — use the config file.
Tuning uses a strict time-based 70/30 split; the holdout result may be looked at once
per major model change.

## Operations (paper trading)

Cron on a small VPS drives the paper loop (times UTC, wrapper + crontab lines
in `scripts/cron.sh`):

- 12:00 `daily-settle` — sync finals, mark closing lines, settle paper picks,
  print the running paper report
- 14:00 `daily` — sync schedule + pitcher snapshots, pull odds, generate and
  print paper picks
- 16:30, 18:00, 22:00, 00:30 `capture-odds` — odds-only snapshot pulls (no sync, no
  picks)

The capture-odds passes exist for CLV integrity: with only the 14:00 pull,
`mark_closing_lines` would flag the very snapshot the picks were priced from
as the closing line, making paper CLV 0 by construction and the go/no-go gate
(avg CLV >= +1.5%) unevaluable. The four extra captures cover early day games
(16:30 — Sunday 13:05 ET starts), later afternoon games, evening ET starts,
and west coast starts. API budget: 5 pulls/day * ~30 days ≈ 150 requests/month
against The Odds API free tier's 500; a paid tier would allow tighter pre-game
captures. The evaluation report tracks the median gap between closing captures
and first pitch — aggregate plus a per-official-date breakdown — and prints a
WARNING when the aggregate median exceeds 120 minutes (stale closes = degraded
CLV quality); a day-game-heavy slate can trip it benignly, which is what the
per-day breakdown is for.

## Commands

- Run CLI: `uv run python -m pickengine --help`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
