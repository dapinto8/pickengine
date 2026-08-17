"""Parameter grid search with a strict time-based train/holdout split.

The grid covers Elo K, pitching Elo-per-FIP, blend weight, and MIN_EV
(4*3*4*4 = 192 combos). The date range is split BY TIME, never randomly:
the earliest ~70% of days is the tuning window, the final ~30% the holdout.
Each combo is backtested on the tuning window only and ranked by average CLV
(Brier and ROI are reported alongside but do not drive the choice). The
winner is then evaluated exactly once on the holdout window.

THE HOLDOUT NUMBER IS THE ONLY ONE THAT COUNTS. The tuning-window figure is
contaminated by the search itself. Look at the holdout at most once per major
model change — re-running tune repeatedly and peeking at holdout results
turns the holdout into a second training set and its number into fiction.

The chosen parameters are written to pickengine.toml (see pickengine.config),
which the pipeline CLI commands load — code constants are never edited.
"""

import json
import uuid
from dataclasses import asdict
from datetime import date, timedelta
from itertools import product
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pickengine.backtest.evaluation import _group_stats, evaluate_run
from pickengine.backtest.runner import (
    DEFAULT_DECISION_LEAD,
    REPORTS_DIR,
    clear_backtest_run,
    run_backtest,
)
from pickengine.config import DEFAULT_CONFIG_PATH, Config, save_config
from pickengine.engine.elo import EloRatings
from pickengine.models import Game, GameStatus, OddsSnapshot, Pick

GRID = {
    "elo_k": [3.0, 4.0, 5.0, 6.0],
    "elo_per_fip": [25.0, 40.0, 55.0],
    "blend_weight_model": [0.2, 0.3, 0.4, 0.5],
    "min_ev": [0.03, 0.04, 0.05, 0.06],
}

TUNE_FRACTION = 0.7

HOLDOUT_WARNING = (
    "WARNING: the holdout result is the ONLY number that counts. The tuning-window "
    "figure is contaminated by the search. Do not re-run tune and peek at the holdout "
    "repeatedly — it may be looked at once per major model change, or it becomes a "
    "second training set."
)


def time_split(start: date, end: date, fraction: float = TUNE_FRACTION) -> tuple[date, date]:
    """(tune_end, holdout_start): earliest `fraction` of days tunes, rest holds out."""
    total_days = (end - start).days + 1
    tune_days = int(total_days * fraction)
    if tune_days < 1 or tune_days >= total_days:
        raise ValueError(f"range {start}..{end} too short for a {fraction:.0%} time split")
    tune_end = start + timedelta(days=tune_days - 1)
    return tune_end, tune_end + timedelta(days=1)


def run_tuning(
    session: Session,
    start: date,
    end: date,
    decision_lead: timedelta = DEFAULT_DECISION_LEAD,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict:
    """Grid-search on the tuning window, evaluate once on holdout, write config."""
    tune_end, holdout_start = time_split(start, end)

    odds_in_window = session.scalar(
        select(func.count())
        .select_from(OddsSnapshot)
        .join(Game, OddsSnapshot.game_id == Game.id)
        .where(Game.date_utc >= start, Game.date_utc <= tune_end)
    )
    if not odds_in_window:
        raise RuntimeError(
            f"no odds snapshots for games in tuning window {start}..{tune_end} — "
            "CLV-based tuning is impossible; import odds first"
        )

    finals = session.scalars(select(Game).where(Game.status == GameStatus.FINAL)).all()
    stamp = uuid.uuid4().hex[:8]

    results = []
    combo_index = 0
    for k in GRID["elo_k"]:
        elo = EloRatings(finals, k=k)  # shared across all combos with this K
        for fip, weight, min_ev in product(
            GRID["elo_per_fip"], GRID["blend_weight_model"], GRID["min_ev"]
        ):
            config = Config(
                elo_k=k, elo_per_fip=fip, blend_weight_model=weight, min_ev=min_ev
            )
            run_id = f"tn{stamp}{combo_index:03d}"
            combo_index += 1
            run_backtest(
                session, start, tune_end, decision_lead,
                run_id=run_id, config=config, elo=elo, write_meta=False,
            )
            picks = session.scalars(select(Pick).where(Pick.run_id == run_id)).all()
            stats = _group_stats(picks)
            clear_backtest_run(session, run_id)
            results.append({"params": asdict(config), **stats})

    # Rank by avg CLV; among ties prefer more picks (more reliable estimate),
    # then the parameter tuple for determinism. Combos with no CLV rank last.
    def rank_key(row: dict) -> tuple:
        clv = row["avg_clv_pct"]
        return (
            -(clv if clv is not None else float("-inf")),
            -row["n"],
            tuple(row["params"].values()),
        )

    results.sort(key=rank_key)
    best = results[0]
    if best["avg_clv_pct"] is None:
        raise RuntimeError(
            "no parameter combination produced picks with closing-line CLV on the "
            "tuning window — nothing to optimize"
        )
    best_config = Config(**best["params"])

    # Full report for the winner on the tuning window, then ONE holdout run.
    tune_run = f"tuned-{stamp}"
    run_backtest(session, start, tune_end, decision_lead, run_id=tune_run, config=best_config)
    tune_report = evaluate_run(session, tune_run, start, tune_end, decision_lead, best_config)

    holdout_run = f"holdout-{stamp}"
    run_backtest(session, holdout_start, end, decision_lead, run_id=holdout_run,
                 config=best_config)
    holdout_report = evaluate_run(
        session, holdout_run, holdout_start, end, decision_lead, best_config
    )

    written_to = save_config(best_config, config_path)
    report = {
        "grid_size": len(results),
        "tune_window": {"start": start.isoformat(), "end": tune_end.isoformat()},
        "holdout_window": {"start": holdout_start.isoformat(), "end": end.isoformat()},
        "chosen_params": asdict(best_config),
        "config_written_to": str(written_to),
        "top_combos": results[:5],
        "tuning_result": tune_report,
        "holdout_result": holdout_report,
        "warning": HOLDOUT_WARNING,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"tune_{stamp}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def render_tuning_report(report: dict) -> str:
    """Console rendering of a run_tuning report."""

    def window_line(label: str, result: dict) -> str:
        picks = result["picks"]
        clv = result["clv"]
        return (
            f"{label}: {picks['n']} picks {picks['wins']}-{picks['losses']}"
            f"-{picks['pushes']}  "
            f"avg CLV {clv['avg_pct']:+.2f}% (beat close {clv['beat_close_pct']}%)  "
            f"ROI {picks['roi_pct']}%  Brier {result['model']['brier_score']}"
            if clv
            else f"{label}: {picks['n']} picks — no CLV data"
        )

    chosen = report["chosen_params"]
    lines = [
        f"Grid search: {report['grid_size']} combos on tuning window "
        f"{report['tune_window']['start']}..{report['tune_window']['end']} "
        f"(earliest {TUNE_FRACTION:.0%} of days; holdout is the rest)",
        "",
        "Top combos by tuning-window avg CLV:",
    ]
    for row in report["top_combos"]:
        p = row["params"]
        lines.append(
            f"  K={p['elo_k']:g} fip={p['elo_per_fip']:g} w={p['blend_weight_model']:g} "
            f"min_ev={p['min_ev']:g}  ->  n={row['n']} "
            f"CLV {row['avg_clv_pct']}% ROI {row['roi_pct']}%"
        )
    lines += [
        "",
        f"Chosen: elo_k={chosen['elo_k']:g}  elo_per_fip={chosen['elo_per_fip']:g}  "
        f"blend_weight_model={chosen['blend_weight_model']:g}  min_ev={chosen['min_ev']:g}",
        f"Written to {report['config_written_to']}",
        "",
        window_line("TUNING ", report["tuning_result"]),
        window_line("HOLDOUT", report["holdout_result"]),
        "",
        report["warning"],
    ]
    return "\n".join(lines)
