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


@app.command()
def initdb() -> None:
    """Create the database schema (idempotent).

    Uses the SQLite file from PICKENGINE_DB, defaulting to ./pickengine.db.
    """
    from pickengine.db import create_schema, get_engine, resolve_db_path

    create_schema(get_engine())
    typer.echo(f"Schema created in {resolve_db_path()}")


@app.command("sync-teams")
def sync_teams_cmd() -> None:
    """Upsert all MLB teams from the StatsAPI."""
    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.ingest.mlb import StatsApiClient, sync_teams

    engine = get_engine()
    create_schema(engine)
    with session_scope(engine) as session:
        count = sync_teams(session, StatsApiClient())
    typer.echo(f"Upserted {count} teams")


@app.command("sync-schedule")
def sync_schedule_cmd(
    start: str = typer.Option(..., help="Start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date, YYYY-MM-DD (inclusive)"),
) -> None:
    """Upsert games (with probable starters and final scores) in a date range."""
    from datetime import date

    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.ingest.mlb import StatsApiClient, sync_schedule

    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    if start_date > end_date:
        raise typer.BadParameter("--start must be on or before --end")
    engine = get_engine()
    create_schema(engine)
    with session_scope(engine) as session:
        counts = sync_schedule(session, StatsApiClient(), start_date, end_date)
    typer.echo(
        f"Upserted {counts['games']} games "
        f"({counts['postponed']} postponed, "
        f"{counts['missing_probables']} team-games missing probable starter, "
        f"{counts['skipped_game_type']} non-regular/postseason skipped)"
    )


@app.command("capture-odds")
def capture_odds_cmd() -> None:
    """Pull current MLB odds (h2h, totals, runline) and store the snapshots.

    Does nothing else — no schedule sync, no pick generation — so it is cheap
    (one API request) and safe to run many times a day. Repeated intra-day
    captures are what make paper CLV real: without them the closing line
    would just be the pick-time snapshot re-flagged, and CLV would be 0 by
    construction. Cron runs this at 18:00, 22:00, and 00:30 UTC on top of
    the 14:00 pull inside `daily` (see scripts/cron.sh).
    """
    from datetime import UTC, datetime

    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.ingest.odds import fetch_live_odds, ingest_live_events, require_api_key

    events = fetch_live_odds(require_api_key())
    engine = get_engine()
    create_schema(engine)
    captured_at = datetime.now(UTC).replace(tzinfo=None)
    with session_scope(engine) as session:
        counts = ingest_live_events(session, events, captured_at)
    typer.echo(_odds_counts_line(counts))


@app.command("import-odds")
def import_odds_cmd(
    path: str = typer.Argument(..., help="CSV or JSON odds dump (schema: see ingest/odds.py)"),
) -> None:
    """Import a historical odds dump from a CSV or JSON file."""
    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.ingest.odds import import_odds_file

    engine = get_engine()
    create_schema(engine)
    with session_scope(engine) as session:
        counts = import_odds_file(session, path)
    typer.echo(_odds_counts_line(counts))
    if counts["ambiguous_doubleheader"]:
        typer.echo(
            "Note: doubleheader rows are recoverable — add an optional commence_time "
            "column (scheduled start, ISO-8601) to the dataset and re-import."
        )
    if counts["inserted"]:
        typer.echo(
            "Note: the daily cron only re-derives closing flags for the last two days — "
            "run `pickengine mark-closing` to flag closing lines for imported dates."
        )


@app.command("mark-closing")
def mark_closing_cmd() -> None:
    """Flag the latest pre-first-pitch snapshot per game/market/book/outcome/line."""
    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.ingest.odds import mark_closing_lines

    engine = get_engine()
    create_schema(engine)
    with session_scope(engine) as session:
        marked = mark_closing_lines(session)
    typer.echo(f"Flagged {marked} closing snapshots")


def _odds_counts_line(counts: dict[str, int]) -> str:
    return (
        f"Inserted {counts['inserted']} odds snapshots "
        f"({counts['duplicates']} duplicates, {counts['unmatched_team']} unmatched teams, "
        f"{counts['unmatched_game']} unmatched games, "
        f"{counts['ambiguous_game']} ambiguous games, "
        f"{counts['ambiguous_doubleheader']} doubleheader rows without commence_time, "
        f"{counts['skipped_market']} skipped markets)"
    )


@app.command("generate-picks")
def generate_picks_cmd(
    date_str: str = typer.Option(..., "--date", help="Pick date, YYYY-MM-DD (MLB official date)"),
    phase: str = typer.Option("paper", help="backtest | paper | live"),
) -> None:
    """Generate, persist, and print the day's picks (h2h only for now)."""
    from datetime import UTC, date, datetime

    from sqlalchemy import select

    from pickengine.db import get_engine, session_scope
    from pickengine.engine.elo import EloRatings
    from pickengine.engine.selection import generate_picks
    from pickengine.models import Game, GameStatus, Phase

    target_date = date.fromisoformat(date_str)
    try:
        phase_enum = Phase(phase.lower())
    except ValueError as exc:
        raise typer.BadParameter(f"phase must be backtest, paper, or live, got {phase!r}") from exc

    from pickengine.config import load_config

    config = load_config()
    as_of = datetime.now(UTC).replace(tzinfo=None)
    engine = get_engine()
    with session_scope(engine) as session:
        finals = session.scalars(select(Game).where(Game.status == GameStatus.FINAL)).all()
        picks = generate_picks(
            session, EloRatings(finals, k=config.elo_k), target_date, phase_enum, as_of,
            min_ev=config.min_ev, blend_weight=config.blend_weight_model,
            elo_per_fip=config.elo_per_fip, devig_method=config.devig_method,
        )
        _print_pick_card(session, picks, f"{target_date} (phase={phase_enum.value})")


def _print_pick_card(session, picks: list, label: str) -> None:
    from sqlalchemy import select

    from pickengine.models import Game, Team

    if not picks:
        typer.echo(f"No qualifying picks for {label}")
        return
    name_by_team = {t.id: t.name for t in session.scalars(select(Team))}
    games = {
        g.id: g
        for g in session.scalars(select(Game).where(Game.id.in_([p.game_id for p in picks])))
    }
    typer.echo(f"Picks for {label}:")
    for p in picks:
        game = games[p.game_id]
        matchup = f"{name_by_team[game.away_team_id]} @ {name_by_team[game.home_team_id]}"
        typer.echo(
            f"  {matchup:45s} side={p.outcome_label:22s} "
            f"odds={p.decimal_odds_at_pick:.2f} book={p.book:12s} "
            f"p_model={p.model_probability:.3f} "
            f"p_market={p.market_consensus_probability:.3f} ev={p.ev:+.1%}"
        )


@app.command()
def settle(
    date_str: str = typer.Option(
        ..., "--date", help="Game date to settle, YYYY-MM-DD (MLB official date)"
    ),
) -> None:
    """Resolve pending picks for a date and fill closing odds + CLV."""
    from datetime import date

    from pickengine.db import get_engine, session_scope
    from pickengine.engine.selection import settle_picks

    with session_scope(get_engine()) as session:
        counts = settle_picks(session, date.fromisoformat(date_str))
    typer.echo(
        f"Settled: {counts['won']} won, {counts['lost']} lost, {counts['push']} push, "
        f"{counts['void']} void; {counts['still_pending']} still pending, "
        f"{counts['no_closing']} without a closing line"
    )


@app.command()
def backtest(
    start: str = typer.Option(..., help="Start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date, YYYY-MM-DD (inclusive)"),
    decision_hours: float = typer.Option(4.0, help="Decision time: hours before first pitch"),
    run_id: str = typer.Option(None, "--run-id", help="Tag for this run (default: new uuid)"),
) -> None:
    """Replay the daily pipeline over a date range with phase=backtest."""
    from datetime import date, timedelta

    from pickengine.backtest.runner import run_backtest
    from pickengine.config import load_config
    from pickengine.db import create_schema, get_engine, session_scope

    engine = get_engine()
    create_schema(engine)
    with session_scope(engine) as session:
        summary = run_backtest(
            session,
            date.fromisoformat(start),
            date.fromisoformat(end),
            decision_lead=timedelta(hours=decision_hours),
            run_id=run_id,
            config=load_config(),
        )
    typer.echo(
        f"Backtest run {summary['run_id']}: {summary['picks']} picks over "
        f"{summary['start']}..{summary['end']} "
        f"(won {summary.get('won', 0)}, lost {summary.get('lost', 0)}, "
        f"push {summary.get('push', 0)}, void {summary.get('void', 0)}, "
        f"{summary.get('no_closing', 0)} without closing line)"
    )
    typer.echo(f"Evaluate with: pickengine evaluate --run-id {summary['run_id']}")


@app.command("clear-backtest")
def clear_backtest_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Backtest run to delete"),
) -> None:
    """Delete all picks belonging to one backtest run."""
    from pickengine.backtest.runner import clear_backtest_run
    from pickengine.db import get_engine, session_scope

    with session_scope(get_engine()) as session:
        deleted = clear_backtest_run(session, run_id)
    typer.echo(f"Deleted {deleted} picks from run {run_id}")


@app.command()
def evaluate(
    run_id: str = typer.Option(..., "--run-id", help="Backtest run to evaluate"),
) -> None:
    """Produce the CLV / calibration / ROI / drawdown report for a run."""
    from pickengine.backtest.evaluation import (
        evaluate_run,
        render_report,
        resolve_run_window,
        write_report,
    )
    from pickengine.config import load_config
    from pickengine.db import get_engine, session_scope

    with session_scope(get_engine()) as session:
        start, end, lead = resolve_run_window(session, run_id)
        report = evaluate_run(session, run_id, start, end, lead, load_config())
    typer.echo(render_report(report))
    typer.echo(f"\nJSON report written to {write_report(report)}")


@app.command()
def tune(
    start: str = typer.Option(..., help="Start date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date, YYYY-MM-DD (inclusive)"),
    decision_hours: float = typer.Option(4.0, help="Decision time: hours before first pitch"),
) -> None:
    """Grid-search model parameters on a time-based 70/30 split.

    Optimizes average CLV on the earliest 70% of days, evaluates the winner
    once on the final 30%, and writes the chosen parameters to
    pickengine.toml.
    """
    from datetime import date, timedelta

    from pickengine.backtest.tuning import render_tuning_report, run_tuning
    from pickengine.db import create_schema, get_engine, session_scope

    engine = get_engine()
    create_schema(engine)
    try:
        with session_scope(engine) as session:
            report = run_tuning(
                session,
                date.fromisoformat(start),
                date.fromisoformat(end),
                decision_lead=timedelta(hours=decision_hours),
            )
    except RuntimeError as exc:
        typer.echo(f"Cannot tune: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(render_tuning_report(report))


@app.command()
def daily() -> None:
    """Morning paper-trading pipeline (cron at 14:00 UTC).

    Syncs the schedule for today and tomorrow, builds pitcher snapshots as of
    today, pulls live odds, and generates paper picks for both dates,
    printing the card. Dates are MLB official dates; generating for (today,
    tomorrow) still covers everything because official dates lag UTC dates,
    never lead them — at 14:00 UTC every game yet to start today or tonight
    carries official date today or tomorrow. Whether a game can still be bet
    is judged only against first_pitch_utc (in generate_picks). Live syncs
    bypass the API file cache so nothing stale leaks in.
    """
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select

    from pickengine.config import load_config
    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.engine.elo import EloRatings
    from pickengine.engine.selection import generate_picks
    from pickengine.ingest.mlb import StatsApiClient, sync_pitcher_snapshots, sync_schedule
    from pickengine.ingest.odds import fetch_live_odds, ingest_live_events, require_api_key
    from pickengine.models import Game, GameStatus, Phase

    api_key = require_api_key()  # fail fast before doing any sync work
    today = datetime.now(UTC).date()
    tomorrow = today + timedelta(days=1)
    client = StatsApiClient(cache_enabled=False)
    engine = get_engine()
    create_schema(engine)

    with session_scope(engine) as session:
        counts = sync_schedule(session, client, today, tomorrow)
    typer.echo(f"Schedule synced: {counts['games']} games ({counts['postponed']} postponed)")

    with session_scope(engine) as session:
        counts = sync_pitcher_snapshots(session, client, today)
    typer.echo(
        f"Pitcher snapshots: {counts['snapshots']} written "
        f"({counts['skipped_no_data']} skipped)"
    )

    events = fetch_live_odds(api_key)
    with session_scope(engine) as session:
        counts = ingest_live_events(session, events, datetime.now(UTC).replace(tzinfo=None))
    typer.echo(_odds_counts_line(counts))

    config = load_config()
    with session_scope(engine) as session:
        finals = session.scalars(select(Game).where(Game.status == GameStatus.FINAL)).all()
        elo = EloRatings(finals, k=config.elo_k)
        picks = []
        for day in (today, tomorrow):
            picks += generate_picks(
                session, elo, day, Phase.PAPER,
                datetime.now(UTC).replace(tzinfo=None),
                min_ev=config.min_ev, blend_weight=config.blend_weight_model,
                elo_per_fip=config.elo_per_fip, devig_method=config.devig_method,
            )
        _print_pick_card(session, picks, f"{today} + {tomorrow} (phase=paper)")


@app.command("daily-settle")
def daily_settle() -> None:
    """Morning-after pipeline (cron at 12:00 UTC).

    Syncs finals for the last two days, marks closing lines, settles paper
    picks for those dates, and prints the running paper-trading report.
    Dates are MLB official dates (official dates lag UTC, never lead, so at
    12:00 UTC yesterday's official slate — including night games that ended
    past midnight UTC — is fully final); sweeping two days back also catches
    late finals and previously postponed settlements.
    """
    from datetime import UTC, datetime, timedelta

    from pickengine.backtest.evaluation import evaluate_phase, render_report, write_report
    from pickengine.config import load_config
    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.engine.selection import settle_picks
    from pickengine.ingest.mlb import StatsApiClient, sync_schedule
    from pickengine.ingest.odds import mark_closing_lines
    from pickengine.models import Phase

    today = datetime.now(UTC).date()
    day_1, day_2 = today - timedelta(days=1), today - timedelta(days=2)
    client = StatsApiClient(cache_enabled=False)
    engine = get_engine()
    create_schema(engine)

    with session_scope(engine) as session:
        counts = sync_schedule(session, client, day_2, day_1)
    typer.echo(f"Finals synced: {counts['games']} games ({counts['postponed']} postponed)")

    with session_scope(engine) as session:
        marked = mark_closing_lines(session, day_2, day_1)
    typer.echo(f"Closing lines flagged: {marked}")

    with session_scope(engine) as session:
        totals = {}
        for day in (day_2, day_1):
            for key, value in settle_picks(session, day, phase=Phase.PAPER).items():
                totals[key] = totals.get(key, 0) + value
    typer.echo(
        f"Settled: {totals['won']} won, {totals['lost']} lost, {totals['push']} push, "
        f"{totals['void']} void; {totals['still_pending']} still pending, "
        f"{totals['no_closing']} without a closing line"
    )

    with session_scope(engine) as session:
        report = evaluate_phase(session, Phase.PAPER, config=load_config())
    if report is None:
        typer.echo("No paper picks yet — nothing to report.")
        return
    typer.echo("")
    typer.echo(render_report(report))
    typer.echo(f"\nJSON report written to {write_report(report)}")


@app.command("export-track-record")
def export_track_record_cmd(
    format: str = typer.Option("md", "--format", help="Output format (md only for now)"),
    out: str = typer.Option("./track_record.md", help="Output file path"),
) -> None:
    """Export the complete, unfiltered paper pick history as a markdown table."""
    from pickengine.db import get_engine, session_scope
    from pickengine.export import export_track_record_md

    if format != "md":
        raise typer.BadParameter(f"only --format md is supported, got {format!r}")
    with session_scope(get_engine()) as session:
        path = export_track_record_md(session, out)
    typer.echo(f"Track record written to {path}")


@app.command("sync-pitchers")
def sync_pitchers_cmd(
    as_of: str = typer.Option(..., "--as-of", help="Snapshot date, YYYY-MM-DD"),
) -> None:
    """Build pitcher stats snapshots as of a date (stats through the prior day)."""
    from datetime import date

    from pickengine.db import create_schema, get_engine, session_scope
    from pickengine.ingest.mlb import StatsApiClient, sync_pitcher_snapshots

    engine = get_engine()
    create_schema(engine)
    with session_scope(engine) as session:
        counts = sync_pitcher_snapshots(session, StatsApiClient(), date.fromisoformat(as_of))
    typer.echo(
        f"Wrote {counts['snapshots']} snapshots "
        f"({counts['skipped_no_data']} pitchers skipped with no prior data)"
    )


def main() -> None:
    """Entrypoint used by `python -m pickengine` and the `pickengine` script."""
    from dotenv import load_dotenv

    # Pick up ODDS_API_KEY / PICKENGINE_DB from ./.env; real environment
    # variables win over .env values (load_dotenv does not override by default).
    load_dotenv()
    app()
