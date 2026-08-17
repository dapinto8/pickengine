"""Smoke test: the package and all skeleton modules import cleanly."""

import importlib

import pickengine

MODULES = [
    "pickengine.cli",
    "pickengine.db",
    "pickengine.models",
    "pickengine.ingest.mlb",
    "pickengine.ingest.odds",
    "pickengine.engine.elo",
    "pickengine.engine.pitching",
    "pickengine.engine.probability",
    "pickengine.engine.devig",
    "pickengine.engine.selection",
    "pickengine.backtest.runner",
    "pickengine.backtest.evaluation",
]


def test_version() -> None:
    assert pickengine.__version__


def test_all_modules_import() -> None:
    for name in MODULES:
        importlib.import_module(name)
