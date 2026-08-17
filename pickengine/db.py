"""Database setup: SQLAlchemy engine, session factory, and schema creation.

SQLite, file-based. Will provide:
- `get_engine(path)` — create/return an Engine for the given SQLite file.
- `get_session(engine)` — session factory / context manager.
- `create_schema(engine)` — create all tables from models.py metadata.

All timestamps are stored in UTC. SQLite has no timezone-aware type, so
datetimes are stored as naive UTC and converted explicitly at the edges.
"""
