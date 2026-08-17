"""Ingestion layer: pulls external data into the database.

Side effects (HTTP, file IO, DB writes) are allowed here and in the CLI
layer only. API errors are never swallowed — fail loudly with context.
"""
