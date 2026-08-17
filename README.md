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
  CSV/JSON dumps (their historical endpoints are paid). Set your API key when live
  ingestion lands.

## Development

```sh
uv run pytest          # run tests
uv run ruff check .    # lint
```

All timestamps are stored in UTC. The hard rule of this codebase: no model input may use
information that was not available before first pitch. See `CLAUDE.md` for full project
conventions.
