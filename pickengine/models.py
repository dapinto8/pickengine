"""SQLAlchemy ORM models.

All datetimes are naive UTC (SQLite has no timezone-aware type). Conversion
happens at the edges (ingestion and CLI), never here. Game.official_date is
the one deliberate exception to UTC: it is MLB's league-assigned local
calendar date, used for identity/grouping only — see its docstring.

Lookahead discipline lives in the timestamp semantics:
- PitcherStatsSnapshot.as_of_date: the snapshot contains stats through the day
  *before* as_of_date, so it may be used to predict games on as_of_date or
  later. A snapshot with as_of_date=2024-06-15 knows nothing about 2024-06-15.
- OddsSnapshot.captured_at_utc: a snapshot captured at or after first pitch is
  never a valid pre-game input.
"""

import enum
from datetime import date, datetime

from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class GameType(enum.Enum):
    REGULAR = "regular"
    POSTSEASON = "postseason"


class GameStatus(enum.Enum):
    SCHEDULED = "scheduled"
    FINAL = "final"
    POSTPONED = "postponed"


class Market(enum.Enum):
    H2H = "h2h"
    TOTALS = "totals"
    RUNLINE = "runline"


class PickStatus(enum.Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    VOID = "void"


class Phase(enum.Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


def _enum(enum_cls: type[enum.Enum]) -> Enum:
    """Store enums as their string values (no native DB enum in SQLite)."""
    return Enum(enum_cls, values_callable=lambda e: [m.value for m in e], native_enum=False)


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_id: Mapped[int] = mapped_column(unique=True)
    abbreviation: Mapped[str] = mapped_column(String(8))
    name: Mapped[str]
    league: Mapped[str] = mapped_column(String(8))  # AL / NL; LVBP later


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_id: Mapped[int] = mapped_column(unique=True)
    name: Mapped[str]
    position: Mapped[str] = mapped_column(String(8))


class Game(Base):
    """One MLB game.

    official_date is MLB's officialDate: the local calendar date the league
    assigns to the game (a late ET night game keeps its ET date even though
    first pitch rolls past midnight UTC). It is the game's identity/grouping
    date — the date used for matching odds rows, CLI --date arguments, the
    daily loops, and reports. first_pitch_utc remains the ONLY field used for
    time-ordering, lookahead checks, and pre-game logic.
    """

    __tablename__ = "games"
    __table_args__ = (Index("ix_games_official_date", "official_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_game_pk: Mapped[int] = mapped_column(unique=True)
    official_date: Mapped[date]
    season: Mapped[int]
    game_type: Mapped[GameType] = mapped_column(_enum(GameType))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_starter_player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    away_starter_player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    home_score: Mapped[int | None]
    away_score: Mapped[int | None]
    status: Mapped[GameStatus] = mapped_column(_enum(GameStatus))
    first_pitch_utc: Mapped[datetime | None]


class PitcherStatsSnapshot(Base):
    """Pitcher stats known *before* as_of_date.

    Contains stats through as_of_date - 1 day, usable to predict games on
    as_of_date or later. The date basis is official dates end to end:
    as_of_date, the game-log split dates it aggregates, and last_start_date
    are all MLB official dates. Never join a snapshot to a game with
    game.official_date < snapshot.as_of_date.
    """

    __tablename__ = "pitcher_stats_snapshots"
    __table_args__ = (
        Index("ix_pitcher_stats_player_asof", "player_id", "as_of_date"),
        UniqueConstraint("player_id", "as_of_date", "season"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    as_of_date: Mapped[date]
    season: Mapped[int]
    ip: Mapped[float] = mapped_column(Float)
    fip: Mapped[float] = mapped_column(Float)
    xfip_proxy: Mapped[float | None] = mapped_column(Float)
    k_per_9: Mapped[float] = mapped_column(Float)
    bb_per_9: Mapped[float] = mapped_column(Float)
    games_started: Mapped[int]
    last_start_date: Mapped[date | None]


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"
    __table_args__ = (
        Index("ix_odds_game_market_captured", "game_id", "market", "captured_at_utc"),
        # Dedup key for ingestion (INSERT .. ON CONFLICT DO NOTHING).
        # line_value is NULL for h2h and SQLite treats NULLs as DISTINCT in
        # unique constraints, so it is coalesced to a sentinel no real
        # total/runline can take — otherwise h2h duplicates would never
        # conflict. Keep in sync with the DDL in db._migrate.
        Index(
            "uq_odds_snapshot_key",
            "game_id", "book", "market", "outcome_label", "captured_at_utc",
            text("coalesce(line_value, -1e9)"),
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    book: Mapped[str]
    market: Mapped[Market] = mapped_column(_enum(Market))
    outcome_label: Mapped[str]
    decimal_odds: Mapped[float] = mapped_column(Float)
    line_value: Mapped[float | None] = mapped_column(Float)  # totals/runline only
    captured_at_utc: Mapped[datetime]
    is_closing: Mapped[bool] = mapped_column(default=False)


class Pick(Base):
    __tablename__ = "picks"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    market: Mapped[Market] = mapped_column(_enum(Market))
    outcome_label: Mapped[str]
    line_value: Mapped[float | None] = mapped_column(Float)
    decimal_odds_at_pick: Mapped[float] = mapped_column(Float)
    book: Mapped[str]
    model_probability: Mapped[float] = mapped_column(Float)
    market_consensus_probability: Mapped[float] = mapped_column(Float)
    ev: Mapped[float] = mapped_column(Float)
    stake_units: Mapped[float] = mapped_column(Float)
    created_at_utc: Mapped[datetime]
    status: Mapped[PickStatus] = mapped_column(_enum(PickStatus), default=PickStatus.PENDING)
    closing_decimal_odds: Mapped[float | None] = mapped_column(Float)  # filled at close
    clv_pct: Mapped[float | None] = mapped_column(Float)  # filled at close
    phase: Mapped[Phase] = mapped_column(_enum(Phase))
    run_id: Mapped[str | None] = mapped_column(String(32))  # backtest run tag; NULL live/paper
