"""Tests for odds ingestion (live + file import), closing lines, and the
no-lookahead odds gate. Fixtures only — no network."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, select

from pickengine.db import create_schema, get_engine, session_scope
from pickengine.ingest.odds import (
    GameIndex,
    get_usable_odds,
    import_odds_file,
    ingest_live_events,
    mark_closing_lines,
    normalize_team_name,
)
from pickengine.models import Game, GameStatus, GameType, Market, OddsSnapshot, Team

FIXTURES = Path(__file__).parent / "fixtures"

FIRST_PITCH = datetime(2024, 6, 15, 23, 10)


@pytest.fixture
def engine() -> Engine:
    engine = get_engine(":memory:")
    create_schema(engine)
    with session_scope(engine) as session:
        mets = Team(mlb_id=121, abbreviation="NYM", name="New York Mets", league="NL")
        braves = Team(mlb_id=144, abbreviation="ATL", name="Atlanta Braves", league="NL")
        yankees = Team(mlb_id=147, abbreviation="NYY", name="New York Yankees", league="AL")
        athletics = Team(mlb_id=133, abbreviation="ATH", name="Athletics", league="AL")
        session.add_all([mets, braves, yankees, athletics])
        session.flush()

        def game(pk: int, d: date, first_pitch: datetime | None) -> Game:
            return Game(
                mlb_game_pk=pk, date_utc=d, season=2024, game_type=GameType.REGULAR,
                home_team_id=mets.id, away_team_id=braves.id,
                status=GameStatus.SCHEDULED, first_pitch_utc=first_pitch,
            )

        session.add_all(
            [
                game(745900, date(2024, 6, 15), FIRST_PITCH),
                # Doubleheader on the 16th:
                game(745901, date(2024, 6, 16), datetime(2024, 6, 16, 17, 10)),
                game(745902, date(2024, 6, 16), datetime(2024, 6, 16, 23, 10)),
            ]
        )
    return engine


def test_team_name_resolution_with_aliases(engine: Engine) -> None:
    with session_scope(engine) as session:
        index = GameIndex(session)
        mets = index.resolve_team("New York Mets")
        assert mets is not None
        assert index.resolve_team("NY Mets") == mets
        assert index.resolve_team("St. Louis Cardinals") is None  # not seeded
        # DB has the 2025 name "Athletics"; historical variants resolve to it.
        assert index.resolve_team("Oakland Athletics") == index.resolve_team("Athletics")
    assert normalize_team_name("St. Louis  Cardinals!") == "st louis cardinals"


def test_ingest_live_events(engine: Engine) -> None:
    events = json.loads((FIXTURES / "odds_api_events.json").read_text(encoding="utf-8"))
    captured_at = datetime(2024, 6, 15, 20, 0)
    with session_scope(engine) as session:
        counts = ingest_live_events(session, events, captured_at)

    assert counts == {
        "inserted": 6,
        "duplicates": 0,
        "unmatched_team": 1,  # Tokyo Giants event
        "unmatched_game": 1,  # Braves-Yankees date with no game in DB
        "ambiguous_game": 0,
        "skipped_market": 1,  # the h2h_lay market
    }

    with session_scope(engine) as session:
        snaps = session.scalars(select(OddsSnapshot)).all()
        assert all(s.book == "pinnacle" and s.captured_at_utc == captured_at for s in snaps)
        game = session.scalars(select(Game).where(Game.mlb_game_pk == 745900)).one()
        assert all(s.game_id == game.id for s in snaps)

        h2h = {s.outcome_label: s for s in snaps if s.market is Market.H2H}
        assert h2h["New York Mets"].decimal_odds == 2.05
        assert h2h["Atlanta Braves"].decimal_odds == 1.87
        assert all(s.line_value is None for s in h2h.values())

        totals = {s.outcome_label: s for s in snaps if s.market is Market.TOTALS}
        assert set(totals) == {"Over", "Under"}
        assert totals["Over"].line_value == 8.5

        runline = {s.outcome_label: s for s in snaps if s.market is Market.RUNLINE}
        assert runline["New York Mets"].line_value == -1.5
        assert runline["Atlanta Braves"].line_value == 1.5


def test_ingest_live_doubleheader_disambiguates_by_time(engine: Engine) -> None:
    event = {
        "commence_time": "2024-06-16T17:10:00Z",
        "home_team": "New York Mets",
        "away_team": "Atlanta Braves",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "New York Mets", "price": 1.95},
                            {"name": "Atlanta Braves", "price": 1.95},
                        ],
                    }
                ],
            }
        ],
    }
    with session_scope(engine) as session:
        counts = ingest_live_events(session, [event], datetime(2024, 6, 16, 15, 0))
        assert counts["inserted"] == 2
        game1 = session.scalars(select(Game).where(Game.mlb_game_pk == 745901)).one()
        assert all(s.game_id == game1.id for s in session.scalars(select(OddsSnapshot)))


def test_import_csv(engine: Engine) -> None:
    with session_scope(engine) as session:
        counts = import_odds_file(session, FIXTURES / "odds_dump.csv")
    assert counts == {
        "inserted": 3,
        "duplicates": 1,  # repeated Braves h2h row
        "unmatched_team": 0,
        "unmatched_game": 1,  # 2024-06-19 has no game
        "ambiguous_game": 0,
        "skipped_market": 0,
    }
    with session_scope(engine) as session:
        snaps = {
            (s.book, s.market, s.outcome_label): s
            for s in session.scalars(select(OddsSnapshot))
        }
        mets = snaps[("pinnacle", Market.H2H, "New York Mets")]
        assert mets.is_closing is True
        assert mets.captured_at_utc == datetime(2024, 6, 15, 22, 55)
        over = snaps[("draftkings", Market.TOTALS, "Over")]
        assert over.is_closing is False  # empty is_closing means false
        assert over.line_value == 8.5

    # Re-importing the same file inserts nothing new.
    with session_scope(engine) as session:
        counts = import_odds_file(session, FIXTURES / "odds_dump.csv")
    assert counts["inserted"] == 0
    assert counts["duplicates"] == 4


def test_import_json(engine: Engine, tmp_path: Path) -> None:
    rows = [
        {
            "date": "2024-06-15",
            "home_team": "NY Mets",
            "away_team": "Atlanta Braves",
            "book": "betmgm",
            "market": "moneyline",
            "outcome": "NY Mets",
            "decimal_odds": 2.0,
            "line": None,
            "timestamp": "2024-06-15T21:00:00Z",
            "is_closing": "false",
        }
    ]
    path = tmp_path / "dump.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with session_scope(engine) as session:
        counts = import_odds_file(session, path)
    assert counts["inserted"] == 1
    with session_scope(engine) as session:
        snap = session.scalars(select(OddsSnapshot)).one()
        assert snap.market is Market.H2H
        assert snap.outcome_label == "New York Mets"  # canonicalized from alias


def test_import_malformed_row_raises_with_row_number(engine: Engine, tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "date,home_team,away_team,book,market,outcome,decimal_odds,line,timestamp,is_closing\n"
        "2024-06-15,New York Mets,Atlanta Braves,pinnacle,h2h,"
        "New York Mets,0.95,,2024-06-15T21:00:00Z,false\n",
        encoding="utf-8",
    )
    with session_scope(engine) as session, pytest.raises(ValueError, match="bad.csv row 2"):
        import_odds_file(session, path)


def _add_snap(session, game_id: int, captured_at: datetime, odds: float,
              is_closing: bool = False, book: str = "pinnacle") -> None:
    session.add(
        OddsSnapshot(
            game_id=game_id, book=book, market=Market.H2H, outcome_label="New York Mets",
            decimal_odds=odds, line_value=None, captured_at_utc=captured_at,
            is_closing=is_closing,
        )
    )


def test_mark_closing_lines(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = session.scalars(select(Game).where(Game.mlb_game_pk == 745900)).one()
        _add_snap(session, game.id, datetime(2024, 6, 15, 20, 0), 2.10, is_closing=True)  # stale
        _add_snap(session, game.id, datetime(2024, 6, 15, 22, 55), 2.02)  # true closer
        _add_snap(session, game.id, datetime(2024, 6, 15, 23, 30), 1.95)  # post-pitch
        _add_snap(session, game.id, datetime(2024, 6, 15, 22, 0), 2.05, book="betmgm")

    with session_scope(engine) as session:
        assert mark_closing_lines(session) == 2  # one per book group

    with session_scope(engine) as session:
        flagged = {
            (s.book, s.captured_at_utc)
            for s in session.scalars(select(OddsSnapshot).where(OddsSnapshot.is_closing))
        }
        assert flagged == {
            ("pinnacle", datetime(2024, 6, 15, 22, 55)),
            ("betmgm", datetime(2024, 6, 15, 22, 0)),
        }


def test_get_usable_odds_enforces_no_lookahead(engine: Engine) -> None:
    with session_scope(engine) as session:
        game = session.scalars(select(Game).where(Game.mlb_game_pk == 745900)).one()
        game_id = game.id
        _add_snap(session, game_id, datetime(2024, 6, 15, 12, 0), 2.10)
        _add_snap(session, game_id, datetime(2024, 6, 15, 22, 55), 2.02)
        _add_snap(session, game_id, datetime(2024, 6, 15, 23, 30), 1.95)  # post-pitch

    with session_scope(engine) as session:
        # At any as_of after the game, the post-pitch snapshot never appears.
        usable = get_usable_odds(session, game_id, as_of=datetime(2024, 6, 16))
        assert [s.captured_at_utc for s in usable] == [
            datetime(2024, 6, 15, 12, 0),
            datetime(2024, 6, 15, 22, 55),
        ]
        # as_of mid-afternoon: only the noon snapshot existed yet.
        usable = get_usable_odds(session, game_id, as_of=datetime(2024, 6, 15, 15, 0))
        assert [s.captured_at_utc for s in usable] == [datetime(2024, 6, 15, 12, 0)]
        # Snapshot captured exactly at as_of is NOT usable (strict <).
        usable = get_usable_odds(session, game_id, as_of=datetime(2024, 6, 15, 12, 0))
        assert usable == []

        with pytest.raises(ValueError, match="no game with id"):
            get_usable_odds(session, 99999, as_of=datetime(2024, 6, 16))

    # A game with unknown first pitch yields nothing, even with snapshots.
    with session_scope(engine) as session:
        game = session.scalars(select(Game).where(Game.mlb_game_pk == 745902)).one()
        game.first_pitch_utc = None
        _add_snap(session, game.id, datetime(2024, 6, 16, 12, 0), 2.0)
        no_pitch_id = game.id
    with session_scope(engine) as session:
        assert get_usable_odds(session, no_pitch_id, as_of=datetime(2024, 6, 17)) == []
