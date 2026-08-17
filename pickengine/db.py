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
from sqlalchemy.orm import Session, sessionmaker

from pickengine.models import Base

DEFAULT_DB_PATH = "./pickengine.db"
_MEMORY = ":memory:"


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
        pick_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(picks)")}
        if "run_id" not in pick_columns:
            conn.exec_driver_sql("ALTER TABLE picks ADD COLUMN run_id VARCHAR(32)")
