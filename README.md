# pickengine

MLB-first sports betting prediction engine and backtester.

Three layers: a de-vigged market baseline from bookmaker odds, an Elo-plus-starting-pitcher
probability model, and an EV screen with hard publishing rules. Evaluated primarily on
closing line value (CLV), then calibration (Brier score), ROI, and max drawdown.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone <repo-url>
cd pickengine
uv sync
```

`uv sync` creates the virtual environment and installs all dependencies (including dev
dependencies pytest and ruff).

## Usage

All functionality goes through a single CLI entrypoint:

```sh
uv run python -m pickengine --help
```

There is no web UI. Commands for ingestion, backtesting, and pick generation will be
added as they are implemented.

### Data sources

- **MLB StatsAPI** (statsapi.mlb.com, free) — schedules, results, starting pitchers, stats.
- **The Odds API** (the-odds-api.com) — live odds; historical snapshots are imported from
  CSV/JSON dumps (their historical endpoints are paid). Put your API key in `.env`
  (copy `.env.example`); the CLI loads it automatically, and real environment
  variables take precedence.

## Daily paper-trading loop

Two commands drive live paper trading, designed for cron on a small VPS:

- `pickengine daily` (14:00 UTC) — sync schedule + pitcher snapshots, pull live
  odds (needs `ODDS_API_KEY`), generate paper picks, print the card.
- `pickengine capture-odds` (18:00, 22:00, 00:30 UTC) — odds-only snapshot
  pulls closer to first pitch, so closing lines are real market closes rather
  than the pick-time snapshot re-flagged (without them paper CLV is 0 by
  construction).
- `pickengine daily-settle` (12:00 UTC) — sync finals, mark closing lines,
  settle paper picks, print the running paper report.

`scripts/cron.sh` wraps both with per-day logs in `./logs/`; the crontab lines
are documented at the top of the script. `pickengine export-track-record`
writes the complete, unfiltered paper pick history to `track_record.md` — the
public, verifiable track record.

## Development

```sh
uv run pytest          # run tests
uv run ruff check .    # lint
```

All timestamps are stored in UTC. The hard rule of this codebase: no model input may use
information that was not available before first pitch. See `CLAUDE.md` for full project
conventions.
