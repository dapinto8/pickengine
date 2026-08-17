"""Public track-record export.

Dumps EVERY paper-phase pick ever made — timestamped, with odds, book, result,
units, and CLV — as one markdown table. No filtering, no omissions: this file
is the verifiable public record backing the future free Telegram channel, and
its credibility depends on losses being as visible as wins.
"""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from pickengine.backtest.evaluation import _pick_outcome
from pickengine.models import Game, Phase, Pick, PickStatus, Team

DEFAULT_TRACK_RECORD_PATH = Path("./track_record.md")


def export_track_record_md(
    session: Session, path: str | Path = DEFAULT_TRACK_RECORD_PATH
) -> Path:
    """Write the full paper track record as markdown. Returns the path."""
    picks = session.scalars(
        select(Pick).where(Pick.phase == Phase.PAPER).order_by(Pick.created_at_utc, Pick.id)
    ).all()
    games = {g.id: g for g in session.scalars(select(Game))}
    names = {t.id: t.name for t in session.scalars(select(Team))}

    settled = [p for p in picks if p.status in (PickStatus.WON, PickStatus.LOST, PickStatus.PUSH)]
    staked = sum(_pick_outcome(p).staked for p in settled)
    profit = sum(_pick_outcome(p).profit for p in settled)
    wins = sum(p.status is PickStatus.WON for p in picks)
    losses = sum(p.status is PickStatus.LOST for p in picks)
    pushes = sum(p.status is PickStatus.PUSH for p in picks)
    clvs = [p.clv_pct for p in picks if p.clv_pct is not None]

    lines = [
        "# pickengine paper-trading track record",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC. "
        "Every paper-phase pick ever made is listed below in the order it was "
        "published — nothing is filtered or removed. All times UTC; stakes in "
        "flat units; CLV compares the odds taken to the closing line "
        "(positive = beat the close).",
        "",
        f"**Record:** {wins}-{losses}-{pushes} · "
        f"**Staked:** {staked:.1f}u · **Profit:** {profit:+.2f}u · "
        f"**ROI:** {(profit / staked * 100):+.1f}%" if staked else
        f"**Record:** {wins}-{losses}-{pushes} · no settled picks yet",
        f"**Avg CLV:** {sum(clvs) / len(clvs):+.2f}% over {len(clvs)} picks with a "
        "closing line" if clvs else "**Avg CLV:** n/a (no closing lines recorded yet)",
        "",
        "| # | Picked (UTC) | Game date | Matchup | Pick | Market | Odds | Book "
        "| Stake | Result | Units | CLV% |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, pick in enumerate(picks, start=1):
        game = games[pick.game_id]
        matchup = f"{names[game.away_team_id]} @ {names[game.home_team_id]}"
        units = _pick_outcome(pick).profit
        units_cell = f"{units:+.2f}" if pick.status is not PickStatus.PENDING else "—"
        clv_cell = f"{pick.clv_pct:+.1f}" if pick.clv_pct is not None else "—"
        line_suffix = f" {pick.line_value:+g}" if pick.line_value is not None else ""
        lines.append(
            f"| {i} | {pick.created_at_utc.strftime('%Y-%m-%d %H:%M')} "
            f"| {game.date_utc.isoformat()} | {matchup} "
            f"| {pick.outcome_label}{line_suffix} | {pick.market.value} "
            f"| {pick.decimal_odds_at_pick:.2f} | {pick.book} "
            f"| {pick.stake_units:g} | {pick.status.value} | {units_cell} | {clv_cell} |"
        )
    if not picks:
        lines.append("| — | no paper picks yet | | | | | | | | | | |")

    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
