"""CLI entrypoint for pickengine.

Single Typer app exposing all commands, e.g.:

    python -m pickengine ingest-schedule --season 2025
    python -m pickengine ingest-odds --file odds_dump.csv
    python -m pickengine backtest --start 2024-04-01 --end 2024-09-30
    python -m pickengine picks --date 2026-08-17

This layer is one of the two places side effects are allowed (the other is
ingestion). Commands parse arguments, open DB sessions, call pure functions
from engine/ and backtest/, and print results. No business logic lives here.
"""

import typer

app = typer.Typer(help="MLB betting prediction engine and backtester.")


@app.callback()
def callback() -> None:
    """MLB betting prediction engine and backtester."""


@app.command()
def version() -> None:
    """Print the pickengine version."""
    from pickengine import __version__

    typer.echo(__version__)


def main() -> None:
    """Entrypoint used by `python -m pickengine` and the `pickengine` script."""
    app()
