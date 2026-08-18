"""Backtest runner: chronological replay of the full daily pipeline.

For each date in range, in order, it simulates what the pipeline would have
done that day: Elo ratings and pitcher snapshots as of each game's decision
time, odds usable at that time (each book's latest quote at or before
decision time — never anything captured later), selection under the hard
publishing rules, then immediate settlement from known finals with CLV
against closing snapshots.

Decision time per game = first_pitch_utc - decision_lead (default 4 hours),
simulating when we would realistically publish.

Guard rails:
- `_assert_pick_integrity` re-verifies, for every pick created, that each
  input carried a timestamp strictly before use: decision time before first
  pitch, the priced quote captured strictly before decision time, the
  starter snapshot the prediction would use checked against an INDEPENDENT
  raw query (the helper's choice must equal the latest snapshot dated at or
  before min(official date, decision date) — re-checking the helper's own
  filter would be tautological), and the Elo update actually used available
  strictly before decision time. Any
  violation raises RuntimeError — the run does not continue on leaked data.
- Every run is tagged with a run_id (uuid) stored on its picks, so parameter
  sweeps never collide; `pickengine clear-backtest --run-id X` deletes one
  run. Run metadata (range, decision lead) is written to
  ./reports/backtest_<run_id>_meta.json for `pickengine evaluate`.
"""

import json
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pickengine.config import Config
from pickengine.engine.elo import EloRatings
from pickengine.engine.probability import starter_snapshot
from pickengine.engine.selection import generate_picks, settle_picks
from pickengine.models import Game, GameStatus, OddsSnapshot, Phase, Pick, PitcherStatsSnapshot

DEFAULT_DECISION_LEAD = timedelta(hours=4)
REPORTS_DIR = Path("./reports")


def run_backtest(
    session: Session,
    start: date,
    end: date,
    decision_lead: timedelta = DEFAULT_DECISION_LEAD,
    run_id: str | None = None,
    config: Config | None = None,
    elo: EloRatings | None = None,
    write_meta: bool = True,
) -> dict:
    """Replay [start, end] day by day (MLB official dates). Returns a
    summary dict (incl. run_id).

    `config` supplies tunable parameters (defaults otherwise); `elo` lets a
    caller reuse a prebuilt EloRatings (its k must match config.elo_k — the
    tuner shares one per K value across combos); `write_meta=False` skips the
    reports/ metadata file (used for throwaway tuning runs).
    """
    if start > end:
        raise ValueError("start must be on or before end")
    run_id = run_id or uuid.uuid4().hex[:12]
    config = config or Config()

    if elo is None:
        finals = session.scalars(select(Game).where(Game.status == GameStatus.FINAL)).all()
        elo = EloRatings(finals, k=config.elo_k)  # get_rating slices by as_of: safe to prebuild
    elif elo.k != config.elo_k:
        raise ValueError(f"prebuilt EloRatings has k={elo.k}, config wants {config.elo_k}")

    picks_created = 0
    settle_totals: Counter[str] = Counter()
    day = start
    while day <= end:
        picks = generate_picks(
            session, elo, day, Phase.BACKTEST, decision_lead=decision_lead, run_id=run_id,
            min_ev=config.min_ev, blend_weight=config.blend_weight_model,
            elo_per_fip=config.elo_per_fip, devig_method=config.devig_method,
        )
        for pick in picks:
            _assert_pick_integrity(session, elo, pick)
        picks_created += len(picks)
        settle_totals.update(settle_picks(session, day, phase=Phase.BACKTEST, run_id=run_id))
        day += timedelta(days=1)

    summary = {
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "decision_lead_hours": decision_lead.total_seconds() / 3600,
        "config": asdict(config),
        "picks": picks_created,
        **dict(settle_totals),
    }
    if write_meta:
        _write_meta(summary)
    return summary


def _write_meta(summary: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"backtest_{summary['run_id']}_meta.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _assert_pick_integrity(session: Session, elo: EloRatings, pick: Pick) -> None:
    """Raise RuntimeError if any input behind this pick post-dates its use."""
    game = session.get(Game, pick.game_id)
    decision = pick.created_at_utc

    def violation(detail: str) -> RuntimeError:
        return RuntimeError(
            f"lookahead violation on pick game_id={pick.game_id} "
            f"({pick.outcome_label}): {detail}"
        )

    if game.first_pitch_utc is None or decision >= game.first_pitch_utc:
        raise violation(f"decision time {decision} not before first pitch {game.first_pitch_utc}")

    quote = session.scalars(
        select(OddsSnapshot).where(
            OddsSnapshot.game_id == game.id,
            OddsSnapshot.book == pick.book,
            OddsSnapshot.market == pick.market,
            OddsSnapshot.outcome_label == pick.outcome_label,
            OddsSnapshot.decimal_odds == pick.decimal_odds_at_pick,
            OddsSnapshot.captured_at_utc < decision,
            OddsSnapshot.captured_at_utc < game.first_pitch_utc,
        )
    ).first()
    if quote is None:
        raise violation(
            f"no snapshot at {pick.book} {pick.decimal_odds_at_pick} captured before {decision}"
        )

    cutoff = min(game.official_date, decision.date())
    for starter_id in (game.home_starter_player_id, game.away_starter_player_id):
        if starter_id is None:
            continue
        chosen = starter_snapshot(session, starter_id, game.official_date, decision)
        # Independent oracle: a raw aggregate, deliberately NOT reusing the
        # helper — re-running starter_snapshot and re-checking its own filter
        # would be tautological and could never catch a broken helper. The
        # latest snapshot legally usable at decision time is the one with the
        # max as_of_date at or before the cutoff; the helper must agree
        # exactly (this single equality also catches a helper that returns a
        # snapshot dated after the cutoff, since the oracle never can).
        allowed_date = session.scalar(
            select(func.max(PitcherStatsSnapshot.as_of_date)).where(
                PitcherStatsSnapshot.player_id == starter_id,
                PitcherStatsSnapshot.as_of_date <= cutoff,
            )
        )
        chosen_date = chosen.as_of_date if chosen is not None else None
        if chosen_date != allowed_date:
            raise violation(
                f"pitcher snapshot mismatch for player {starter_id}: "
                f"starter_snapshot chose {chosen_date} at decision {decision} "
                f"but the latest snapshot at or before cutoff {cutoff} is "
                f"{allowed_date}"
            )

    for team_id in (game.home_team_id, game.away_team_id):
        last_update = elo.last_update_time(team_id, decision)
        if last_update is not None and last_update >= decision:
            raise violation(f"elo update available {last_update} used at decision {decision}")


def clear_backtest_run(session: Session, run_id: str) -> int:
    """Delete all picks belonging to one backtest run. Returns rows deleted."""
    picks = session.scalars(select(Pick).where(Pick.run_id == run_id)).all()
    for pick in picks:
        session.delete(pick)
    return len(picks)
