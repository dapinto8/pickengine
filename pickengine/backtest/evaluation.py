"""Backtest evaluation: CLV, calibration (Brier), ROI, drawdown.

Metric priority (per CLAUDE.md): CLV first — beating the closing line is the
only signal that survives small samples; calibration second; ROI and max
drawdown informational until the pick count is large.

CLV convention (set at settlement): clv_pct = (odds_at_pick / closing_odds
- 1) * 100, positive = we beat the close.

The Brier score and calibration table are computed over ALL final games in
the run's range — not just picked ones — by re-running predict_game at each
game's decision time. When no odds exist, p_blend falls back to the pure
model, and the report flags how many games actually had a market.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from pickengine.config import Config
from pickengine.engine.elo import EloRatings
from pickengine.engine.probability import predict_game
from pickengine.engine.selection import _closing_snapshot
from pickengine.models import Game, GameStatus, Phase, Pick, PickStatus

REPORTS_DIR = Path("./reports")

# Closing-capture data-quality guard: when the median gap between the closing
# snapshot's capture time and first pitch exceeds this, "closing" odds are too
# stale to trust — likely just the pick-time snapshot re-flagged — and the
# report carries a warning instead of failing silently with CLV ~ 0.
STALE_CLOSING_GAP_MINUTES = 120.0


def brier_score(pairs: list[tuple[float, int]]) -> float:
    """Mean squared error of (probability, outcome in {0,1}) pairs."""
    if not pairs:
        raise ValueError("brier_score needs at least one prediction")
    return sum((p - outcome) ** 2 for p, outcome in pairs) / len(pairs)


def calibration_table(pairs: list[tuple[float, int]]) -> list[dict]:
    """10%-bucket calibration: predicted bucket, count, avg p, actual rate."""
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, outcome in pairs:
        buckets[min(int(p * 10), 9)].append((p, outcome))
    table = []
    for idx in sorted(buckets):
        rows = buckets[idx]
        table.append(
            {
                "bucket": f"{idx * 10}-{idx * 10 + 10}%",
                "n": len(rows),
                "avg_predicted": sum(p for p, _ in rows) / len(rows),
                "actual_rate": sum(o for _, o in rows) / len(rows),
            }
        )
    return table


def max_drawdown(profits: list[float]) -> float:
    """Largest peak-to-trough decline of the cumulative profit curve."""
    peak = cumulative = 0.0
    drawdown = 0.0
    for profit in profits:
        cumulative += profit
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


@dataclass
class _PickOutcome:
    profit: float
    staked: float
    returned: float


def _pick_outcome(pick: Pick) -> _PickOutcome:
    if pick.status is PickStatus.WON:
        return _PickOutcome(
            pick.stake_units * (pick.decimal_odds_at_pick - 1),
            pick.stake_units,
            pick.stake_units * pick.decimal_odds_at_pick,
        )
    if pick.status is PickStatus.LOST:
        return _PickOutcome(-pick.stake_units, pick.stake_units, 0.0)
    if pick.status is PickStatus.PUSH:
        return _PickOutcome(0.0, pick.stake_units, pick.stake_units)
    return _PickOutcome(0.0, 0.0, 0.0)  # void / pending: stake never at risk


def _group_stats(picks: list[Pick]) -> dict:
    settled = [p for p in picks if p.status in (PickStatus.WON, PickStatus.LOST, PickStatus.PUSH)]
    staked = sum(_pick_outcome(p).staked for p in settled)
    profit = sum(_pick_outcome(p).profit for p in settled)
    clvs = [p.clv_pct for p in picks if p.clv_pct is not None]
    return {
        "n": len(picks),
        "wins": sum(p.status is PickStatus.WON for p in picks),
        "losses": sum(p.status is PickStatus.LOST for p in picks),
        "profit_units": round(profit, 3),
        "roi_pct": round(profit / staked * 100, 2) if staked else None,
        "avg_clv_pct": round(sum(clvs) / len(clvs), 3) if clvs else None,
    }


def evaluate_run(
    session: Session,
    run_id: str,
    start: date,
    end: date,
    decision_lead: timedelta,
    config: Config | None = None,
) -> dict:
    """Build the full report dict for one backtest run."""
    picks = session.scalars(
        select(Pick).where(Pick.run_id == run_id).order_by(Pick.created_at_utc, Pick.id)
    ).all()
    return _build_report(session, run_id, picks, start, end, decision_lead, config)


def evaluate_phase(
    session: Session,
    phase: Phase,
    decision_lead: timedelta = timedelta(hours=4),
    config: Config | None = None,
) -> dict | None:
    """Running report over ALL picks of a phase (e.g. the paper track record).

    The window is derived from the picked games' dates. Returns None when the
    phase has no picks yet.
    """
    picks = session.scalars(
        select(Pick).where(Pick.phase == phase).order_by(Pick.created_at_utc, Pick.id)
    ).all()
    if not picks:
        return None
    game_dates = session.scalars(
        select(Game.official_date).where(Game.id.in_({p.game_id for p in picks}))
    ).all()
    return _build_report(
        session, f"phase-{phase.value}", picks,
        min(game_dates), max(game_dates), decision_lead, config,
    )


def _build_report(
    session: Session,
    label: str,
    picks: list[Pick],
    start: date,
    end: date,
    decision_lead: timedelta,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    outcomes = [_pick_outcome(p) for p in picks]
    staked = sum(o.staked for o in outcomes)
    returned = sum(o.returned for o in outcomes)
    profit = returned - staked

    clvs = [p.clv_pct for p in picks if p.clv_pct is not None]
    clv_summary = None
    if clvs:
        ordered = sorted(clvs)
        clv_summary = {
            "n": len(clvs),
            "avg_pct": round(sum(clvs) / len(clvs), 3),
            "beat_close_pct": round(sum(c > 0 for c in clvs) / len(clvs) * 100, 1),
            "min": round(ordered[0], 3),
            "p25": round(ordered[len(ordered) // 4], 3),
            "median": round(median(ordered), 3),
            "p75": round(ordered[(3 * len(ordered)) // 4], 3),
            "max": round(ordered[-1], 3),
        }

    gaps = _closing_gap_minutes(session, picks)
    closing_capture = None
    if gaps:
        # Round before comparing so the stored flag can never contradict the
        # stored number (a raw median of 120.04 must not print "120" + WARNING).
        median_gap = round(median(g for _, g in gaps), 1)
        gaps_by_day: dict[date, list[float]] = defaultdict(list)
        for official, gap in gaps:
            gaps_by_day[official].append(gap)
        closing_capture = {
            "n": len(gaps),
            "median_gap_minutes": median_gap,
            "stale": median_gap > STALE_CLOSING_GAP_MINUTES,
            "by_day": [
                {
                    "date": official.isoformat(),
                    "n": len(day_gaps),
                    "median_gap_minutes": round(median(day_gaps), 1),
                }
                for official, day_gaps in sorted(gaps_by_day.items())
            ],
        }

    pairs, with_market = _model_predictions(session, start, end, decision_lead, config)

    by_month: dict[str, list[Pick]] = defaultdict(list)
    for pick in picks:
        by_month[pick.created_at_utc.strftime("%Y-%m")].append(pick)
    favorites = [p for p in picks if p.market_consensus_probability >= 0.5]
    underdogs = [p for p in picks if p.market_consensus_probability < 0.5]

    return {
        "run_id": label,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "decision_lead_hours": decision_lead.total_seconds() / 3600,
        "picks": {
            "n": len(picks),
            "wins": sum(p.status is PickStatus.WON for p in picks),
            "losses": sum(p.status is PickStatus.LOST for p in picks),
            "pushes": sum(p.status is PickStatus.PUSH for p in picks),
            "voids": sum(p.status is PickStatus.VOID for p in picks),
            "pending": sum(p.status is PickStatus.PENDING for p in picks),
            "units_staked": round(staked, 3),
            "units_returned": round(returned, 3),
            "profit_units": round(profit, 3),
            "roi_pct": round(profit / staked * 100, 2) if staked else None,
            "max_drawdown_units": round(max_drawdown([o.profit for o in outcomes]), 3),
        },
        "clv": clv_summary,
        "closing_capture": closing_capture,
        "model": {
            "n_games": len(pairs),
            "n_with_market": with_market,
            "brier_score": round(brier_score(pairs), 5) if pairs else None,
            "calibration": [
                {**row, "avg_predicted": round(row["avg_predicted"], 4),
                 "actual_rate": round(row["actual_rate"], 4)}
                for row in calibration_table(pairs)
            ] if pairs else [],
        },
        "by_month": {month: _group_stats(rows) for month, rows in sorted(by_month.items())},
        "by_side": {
            "favorites": _group_stats(favorites),
            "underdogs": _group_stats(underdogs),
        },
    }


def _closing_gap_minutes(session: Session, picks: list[Pick]) -> list[tuple[date, float]]:
    """(official date, minutes between closing capture and first pitch) per
    settled pick.

    Re-resolves the closing snapshot the same way settlement did
    (selection._closing_snapshot); picks without closing odds, a resolvable
    snapshot, or a known first pitch contribute nothing. The official date
    rides along so the report can break the gaps down per day — a day-game
    slate legitimately closes on an earlier capture than a night slate.
    """
    settled = [p for p in picks if p.closing_decimal_odds is not None]
    if not settled:
        return []
    games = {
        game_id: (official, fp)
        for game_id, official, fp in session.execute(
            select(Game.id, Game.official_date, Game.first_pitch_utc).where(
                Game.id.in_({p.game_id for p in settled})
            )
        )
    }
    gaps = []
    for pick in settled:
        official, fp = games.get(pick.game_id, (None, None))
        if fp is None:
            continue
        snapshot = _closing_snapshot(session, pick)
        if snapshot is None:
            continue
        gaps.append((official, (fp - snapshot.captured_at_utc).total_seconds() / 60))
    return gaps


def _model_predictions(
    session: Session,
    start: date,
    end: date,
    decision_lead: timedelta,
    config: Config | None = None,
) -> tuple[list[tuple[float, int]], int]:
    """(p_blend, home_won) for every final game in range, at decision time."""
    config = config or Config()
    finals = session.scalars(select(Game).where(Game.status == GameStatus.FINAL)).all()
    elo = EloRatings(finals, k=config.elo_k)
    pairs: list[tuple[float, int]] = []
    with_market = 0
    for game in finals:
        if not (start <= game.official_date <= end) or game.first_pitch_utc is None:
            continue
        if game.home_score == game.away_score:
            continue  # no binary outcome to score
        prediction = predict_game(
            session, elo, game.id, game.first_pitch_utc - decision_lead,
            blend_weight=config.blend_weight_model, elo_per_fip=config.elo_per_fip,
            devig_method=config.devig_method,
        )
        if prediction.p_market is not None:
            with_market += 1
        pairs.append((prediction.p_blend, int(game.home_score > game.away_score)))
    return pairs, with_market


def render_report(report: dict) -> str:
    """Human-readable console rendering of an evaluate_run report."""
    p = report["picks"]
    lines = [
        f"Backtest report — run {report['run_id']} "
        f"({report['start']} .. {report['end']}, "
        f"decision {report['decision_lead_hours']:g}h before first pitch)",
        "",
        f"Picks: {p['n']}  W-L-P: {p['wins']}-{p['losses']}-{p['pushes']}  "
        f"voids: {p['voids']}  pending: {p['pending']}",
        f"Units staked: {p['units_staked']}  returned: {p['units_returned']}  "
        f"profit: {p['profit_units']:+}  "
        f"ROI: {p['roi_pct'] if p['roi_pct'] is not None else 'n/a'}%  "
        f"max drawdown: {p['max_drawdown_units']}u",
    ]
    clv = report["clv"]
    if clv:
        lines += [
            "",
            f"CLV (n={clv['n']}): avg {clv['avg_pct']:+.2f}%  "
            f"beat close: {clv['beat_close_pct']}%",
            f"  distribution: min {clv['min']:+.2f}  p25 {clv['p25']:+.2f}  "
            f"median {clv['median']:+.2f}  p75 {clv['p75']:+.2f}  max {clv['max']:+.2f}",
        ]
    else:
        lines += ["", "CLV: no picks with closing lines — not evaluable"]

    capture = report.get("closing_capture")
    if capture:
        lines.append(
            f"  closing captured median {capture['median_gap_minutes']:g} min before "
            f"first pitch (n={capture['n']})"
        )
        if capture["stale"]:
            lines += [
                "",
                f"WARNING: closing captures are stale (median gap > "
                f"{STALE_CLOSING_GAP_MINUTES:g} min before first pitch) — CLV quality is "
                "degraded; the closing line may just be the pick-time snapshot "
                "re-flagged. Check that the capture-odds cron passes are running. "
                "Note: a day-game-heavy slate (early afternoon starts) can also "
                "produce a large median gap with a healthy cron — compare against "
                "the per-day breakdown below before assuming captures are broken.",
            ]
            for day in capture.get("by_day", []):
                lines.append(
                    f"  {day['date']}: median {day['median_gap_minutes']:g} min "
                    f"(n={day['n']})"
                )

    model = report["model"]
    lines += [
        "",
        f"Model calibration over {model['n_games']} final games "
        f"({model['n_with_market']} with market odds): "
        f"Brier {model['brier_score']}",
        f"  {'bucket':>8} {'n':>5} {'avg_p':>7} {'actual':>7}",
    ]
    for row in model["calibration"]:
        lines.append(
            f"  {row['bucket']:>8} {row['n']:>5} "
            f"{row['avg_predicted']:>7.3f} {row['actual_rate']:>7.3f}"
        )

    if report["by_month"]:
        lines += ["", f"  {'month':>8} {'n':>4} {'W-L':>7} {'profit':>8} {'ROI%':>7} {'CLV%':>6}"]
        for month, stats in report["by_month"].items():
            lines.append(
                f"  {month:>8} {stats['n']:>4} "
                f"{str(stats['wins']) + '-' + str(stats['losses']):>7} "
                f"{stats['profit_units']:>+8.2f} "
                f"{stats['roi_pct'] if stats['roi_pct'] is not None else '  n/a':>7} "
                f"{stats['avg_clv_pct'] if stats['avg_clv_pct'] is not None else '  n/a':>6}"
            )
    for side in ("favorites", "underdogs"):
        stats = report["by_side"][side]
        lines.append(
            f"  {side:>10}: n={stats['n']} {stats['wins']}-{stats['losses']} "
            f"profit {stats['profit_units']:+.2f}u "
            f"ROI {stats['roi_pct'] if stats['roi_pct'] is not None else 'n/a'}% "
            f"CLV {stats['avg_clv_pct'] if stats['avg_clv_pct'] is not None else 'n/a'}%"
        )
    return "\n".join(lines)


def write_report(report: dict) -> Path:
    """Persist the report JSON to ./reports/. Returns the path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"backtest_{report['run_id']}_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_run_meta(run_id: str) -> dict | None:
    """Run metadata written by the backtest runner, if present."""
    path = REPORTS_DIR / f"backtest_{run_id}_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_window(
    session: Session, run_id: str
) -> tuple[date, date, timedelta]:
    """Run window from meta file, else from the run's picks; error if neither.

    The pick-derived fallback uses the picked games' OFFICIAL dates — the
    same basis the window is later compared against (_model_predictions) —
    never Pick.created_at_utc.date(): a decision time that crosses UTC
    midnight (late west coast game, short lead) would shift the window a day
    forward and silently drop the first official day from the model metrics.
    """
    meta = load_run_meta(run_id)
    if meta is not None:
        return (
            date.fromisoformat(meta["start"]),
            date.fromisoformat(meta["end"]),
            timedelta(hours=meta["decision_lead_hours"]),
        )
    game_dates = session.scalars(
        select(Game.official_date).join(Pick, Pick.game_id == Game.id).where(
            Pick.run_id == run_id
        )
    ).all()
    if not game_dates:
        raise ValueError(
            f"run {run_id!r} has no metadata file and no picks — nothing to evaluate"
        )
    return min(game_dates), max(game_dates), timedelta(hours=4)
