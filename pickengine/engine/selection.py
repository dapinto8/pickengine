"""Pick selection: EV screening and hard publishing rules, plus settlement.

Pipeline for a date (h2h only for now; totals come later):
1. Every game that day that has not started yet (first_pitch_utc > as_of) and
   has a complete de-vigged market quote gets p_blend for home and away.
2. Each side's EV is computed against the best available decimal odds across
   books (each book's LATEST usable quote — never a stale better price).
3. Hard rules, in order: EV >= MIN_EV; at most one pick per game (no
   correlated outcomes — the higher-EV side wins); if more than
   MAX_PICKS_PER_DAY qualify, keep the highest-EV ones. Flat STAKE_UNITS.
   A game that already has a pick in the same phase is never re-picked, so
   re-running a day is idempotent. No market quote -> no pick: without an
   anchor, p_blend is the raw model and not publishable.

The rules are hard gates: a candidate failing any of them is not published,
ever.

Settlement (`settle_picks`): pending picks on final games resolve to
won/lost/push by score vs outcome_label; picks on postponed games are void.
closing_decimal_odds comes from the is_closing snapshot of the same
market/outcome/line — preferring the pick's own book, then
SHARP_BOOK_PRIORITY, then the latest-captured flag. CLV is
(decimal_odds_at_pick / closing_decimal_odds - 1) * 100: POSITIVE means we
took a better price than the close (we beat the market's final assessment).
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from pickengine.engine.elo import EloRatings
from pickengine.engine.pitching import ELO_PER_FIP
from pickengine.engine.probability import (
    BLEND_WEIGHT_MODEL,
    SHARP_BOOK_PRIORITY,
    latest_h2h_quotes,
    predict_game,
)
from pickengine.ingest.odds import get_usable_odds
from pickengine.models import Game, GameStatus, Market, OddsSnapshot, Phase, Pick, PickStatus, Team

MIN_EV = 0.04
MAX_PICKS_PER_DAY = 4
STAKE_UNITS = 1.0


@dataclass(frozen=True)
class Candidate:
    game_id: int
    outcome_label: str
    decimal_odds: float
    book: str
    p_model: float
    p_market: float
    p_blend: float
    ev: float
    as_of: datetime


def expected_value(probability: float, decimal_odds: float) -> float:
    """EV per unit staked: p * odds - 1."""
    return probability * decimal_odds - 1


def generate_picks(
    session: Session,
    elo: EloRatings,
    target_date: date,
    phase: Phase,
    as_of: datetime | None = None,
    *,
    decision_lead: timedelta | None = None,
    run_id: str | None = None,
    min_ev: float = MIN_EV,
    blend_weight: float = BLEND_WEIGHT_MODEL,
    elo_per_fip: float = ELO_PER_FIP,
) -> list[Pick]:
    """Select and persist picks for one date. Returns the new Pick rows.

    Decision time: pass either one fixed `as_of` for the whole day (live /
    paper use: "now"), or `decision_lead` to evaluate each game at
    first_pitch - lead (backtest use). Exactly one must be given. Each pick's
    created_at_utc records the decision time actually used.

    `run_id` tags backtest picks; the no-re-pick guard is scoped to
    (phase, run_id), so separate runs never collide with each other or with
    untagged paper/live picks.
    """
    if (as_of is None) == (decision_lead is None):
        raise ValueError("provide exactly one of as_of or decision_lead")
    games = session.scalars(select(Game).where(Game.date_utc == target_date)).all()
    run_filter = Pick.run_id == run_id if run_id is not None else Pick.run_id.is_(None)
    already_picked = set(
        session.scalars(
            select(Pick.game_id).where(
                Pick.phase == phase, run_filter, Pick.game_id.in_([g.id for g in games])
            )
        )
    )
    name_by_team = {t.id: t.name for t in session.scalars(select(Team))}

    candidates: list[Candidate] = []
    for game in games:
        if game.id in already_picked:
            continue
        if game.first_pitch_utc is None:
            continue
        game_as_of = as_of if as_of is not None else game.first_pitch_utc - decision_lead
        if game.first_pitch_utc <= game_as_of:
            continue  # can't bet a game that has started
        quotes = latest_h2h_quotes(get_usable_odds(session, game.id, game_as_of))
        if not quotes:
            continue  # no odds at all: skip before doing any model work
        prediction = predict_game(
            session, elo, game.id, game_as_of,
            blend_weight=blend_weight, elo_per_fip=elo_per_fip,
        )
        if prediction.p_market is None:
            continue

        sides = (
            (name_by_team[game.home_team_id], prediction.p_model,
             prediction.p_market, prediction.p_blend),
            (name_by_team[game.away_team_id], 1 - prediction.p_model,
             1 - prediction.p_market, 1 - prediction.p_blend),
        )
        game_best: Candidate | None = None
        for label, p_model, p_market, p_blend in sides:
            side_quotes = [s for (_, lbl), s in quotes.items() if lbl == label]
            if not side_quotes:
                continue
            best = max(side_quotes, key=lambda s: s.decimal_odds)
            ev = expected_value(p_blend, best.decimal_odds)
            if ev < min_ev:
                continue
            if game_best is None or ev > game_best.ev:
                game_best = Candidate(
                    game_id=game.id, outcome_label=label, decimal_odds=best.decimal_odds,
                    book=best.book, p_model=p_model, p_market=p_market,
                    p_blend=p_blend, ev=ev, as_of=game_as_of,
                )
        if game_best is not None:
            candidates.append(game_best)

    selected = sorted(candidates, key=lambda c: (-c.ev, c.game_id, c.outcome_label))
    selected = selected[:MAX_PICKS_PER_DAY]

    picks = [
        Pick(
            game_id=c.game_id, market=Market.H2H, outcome_label=c.outcome_label,
            line_value=None, decimal_odds_at_pick=c.decimal_odds, book=c.book,
            model_probability=c.p_model, market_consensus_probability=c.p_market,
            ev=c.ev, stake_units=STAKE_UNITS, created_at_utc=c.as_of,
            status=PickStatus.PENDING, phase=phase, run_id=run_id,
        )
        for c in selected
    ]
    session.add_all(picks)
    return picks


def settle_picks(
    session: Session,
    target_date: date,
    phase: Phase | None = None,
    run_id: str | None = None,
) -> dict[str, int]:
    """Resolve pending picks for a date; fill closing odds and CLV.

    Optional phase/run_id filters scope settlement (a backtest run settles
    only its own picks, never concurrent paper/live ones).
    """
    query = (
        select(Pick, Game)
        .join(Game, Pick.game_id == Game.id)
        .where(Game.date_utc == target_date, Pick.status == PickStatus.PENDING)
    )
    if phase is not None:
        query = query.where(Pick.phase == phase)
    if run_id is not None:
        query = query.where(Pick.run_id == run_id)
    rows = session.execute(query).all()
    name_by_team = {t.id: t.name for t in session.scalars(select(Team))}

    counts = {"won": 0, "lost": 0, "push": 0, "void": 0, "still_pending": 0, "no_closing": 0}
    for pick, game in rows:
        if game.status is GameStatus.POSTPONED:
            pick.status = PickStatus.VOID
            counts["void"] += 1
            continue  # no meaningful closing line for a game never played
        if game.status is not GameStatus.FINAL:
            counts["still_pending"] += 1
            continue

        if pick.market is not Market.H2H:
            raise NotImplementedError(f"settlement for market {pick.market} not implemented")
        if game.home_score == game.away_score:
            pick.status = PickStatus.PUSH
        else:
            winner_id = (
                game.home_team_id if game.home_score > game.away_score else game.away_team_id
            )
            won = pick.outcome_label == name_by_team[winner_id]
            pick.status = PickStatus.WON if won else PickStatus.LOST
        counts[pick.status.value] += 1

        closing = _closing_snapshot(session, pick)
        if closing is None:
            counts["no_closing"] += 1
        else:
            pick.closing_decimal_odds = closing.decimal_odds
            pick.clv_pct = (pick.decimal_odds_at_pick / closing.decimal_odds - 1) * 100
    return counts


def _closing_snapshot(session: Session, pick: Pick) -> OddsSnapshot | None:
    """The closing quote for a pick's market/outcome/line.

    Book preference: the pick's own book, then SHARP_BOOK_PRIORITY, then
    whichever closing flag was captured latest.
    """
    line_filter = (
        OddsSnapshot.line_value.is_(None)
        if pick.line_value is None
        else OddsSnapshot.line_value == pick.line_value
    )
    closers = session.scalars(
        select(OddsSnapshot).where(
            OddsSnapshot.game_id == pick.game_id,
            OddsSnapshot.market == pick.market,
            OddsSnapshot.outcome_label == pick.outcome_label,
            line_filter,
            OddsSnapshot.is_closing,
        )
    ).all()
    if not closers:
        return None
    by_book = {s.book: s for s in closers}
    for book in (pick.book, *SHARP_BOOK_PRIORITY):
        if book in by_book:
            return by_book[book]
    return max(closers, key=lambda s: s.captured_at_utc)
