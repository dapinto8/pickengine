"""Database setup: SQLAlchemy engine, session factory, and schema creation.

SQLite, file-based. The database path comes from the PICKENGINE_DB environment
variable, defaulting to ./pickengine.db. All datetimes are stored as naive UTC
(SQLite has no timezone-aware type); conversion happens at the edges.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from pickengine.models import Base

DEFAULT_DB_PATH = "./pickengine.db"
_MEMORY = ":memory:"

# Bumped on breaking schema changes that are NOT auto-migrated.
# v2: games.date_utc (UTC calendar date) renamed to games.official_date
#     (MLB's official local date) — a semantic change, so old rows are wrong,
#     not just misnamed; the DB must be rebuilt from source data. The
#     uq_odds_snapshot_key unique index (odds dedup for ON CONFLICT inserts)
#     rides the same drop-and-rebuild — one version bump covers both; on
#     already-rebuilt v2 files _migrate adds the index in place.
SCHEMA_VERSION = 2


def resolve_db_path() -> str:
    """Database file path from PICKENGINE_DB, defaulting to ./pickengine.db."""
    return os.environ.get("PICKENGINE_DB", DEFAULT_DB_PATH)


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Create an Engine for the given SQLite file (or ":memory:").

    With no argument, the path is resolved from the environment via
    `resolve_db_path()`. Foreign key enforcement is switched on for every
    connection (SQLite ships with it off).
    """
    path = str(db_path) if db_path is not None else resolve_db_path()
    url = f"sqlite:///{path}" if path == _MEMORY else f"sqlite:///{Path(path)}"
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to the given engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    session = get_session_factory(engine)()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def create_schema(engine: Engine) -> None:
    """Create all tables (idempotent), applying additive column migrations."""
    Base.metadata.create_all(engine)
    _migrate(engine)


def _migrate(engine: Engine) -> None:
    """Minimal additive migrations for pre-existing SQLite files.

    create_all only creates missing tables; columns added to models later
    must be ALTERed in here. Keep entries append-only.
    """
    with engine.begin() as conn:
        game_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(games)")}
        if "date_utc" in game_columns:
            raise RuntimeError(
                f"this database predates schema version {SCHEMA_VERSION}: games.date_utc "
                "was replaced by games.official_date (MLB's official local date), and the "
                "old UTC-derived values are semantically wrong for late games. Delete the "
                "DB file and rebuild: pickengine initdb, then sync-teams / sync-schedule / "
                "sync-pitchers (and re-import odds)."
            )
        pick_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(picks)")}
        if "run_id" not in pick_columns:
            conn.exec_driver_sql("ALTER TABLE picks ADD COLUMN run_id VARCHAR(32)")
        # create_all does not add indexes to pre-existing tables. Keep this
        # DDL in sync with uq_odds_snapshot_key in models.OddsSnapshot.
        # (Deliberately raw, not generated from the model Index: SQLite
        # cannot reflect expression indexes, so index.create(checkfirst=True)
        # raises "already exists" on every run after the first.)
        try:
            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_odds_snapshot_key ON odds_snapshots "
                "(game_id, book, market, outcome_label, captured_at_utc, "
                "coalesce(line_value, -1e9))"
            )
        except IntegrityError as exc:
            # Without the guard this bricks EVERY command (they all call
            # create_schema first) with a bare UNIQUE-constraint error.
            raise RuntimeError(
                "cannot create the odds dedup index uq_odds_snapshot_key: "
                "odds_snapshots already contains duplicate rows (same game/book/"
                "market/outcome/line/captured_at). Delete the duplicates keeping one "
                "row per key, or rebuild the DB from source data, then rerun."
            ) from exc
