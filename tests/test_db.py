"""Schema round-trip: insert one row per table into an in-memory DB, read back."""

from datetime import date, datetime

import pytest
from sqlalchemy import Engine, select

from pickengine.db import create_schema, get_engine, session_scope
from pickengine.models import (
    Game,
    GameStatus,
    GameType,
    Market,
    OddsSnapshot,
    Phase,
    Pick,
    PickStatus,
    PitcherStatsSnapshot,
    Player,
    Team,
)


@pytest.fixture
def engine() -> Engine:
    engine = get_engine(":memory:")
    create_schema(engine)
    return engine


def test_round_trip_all_tables(engine: Engine) -> None:
    with session_scope(engine) as session:
        home = Team(mlb_id=121, abbreviation="NYM", name="New York Mets", league="NL")
        away = Team(mlb_id=144, abbreviation="ATL", name="Atlanta Braves", league="NL")
        pitcher = Player(mlb_id=656849, name="David Peterson", position="P")
        session.add_all([home, away, pitcher])
        session.flush()

        game = Game(
            mlb_game_pk=745123,
            date_utc=date(2024, 6, 15),
            season=2024,
            game_type=GameType.REGULAR,
            home_team_id=home.id,
            away_team_id=away.id,
            home_starter_player_id=pitcher.id,
            away_starter_player_id=None,
            home_score=None,
            away_score=None,
            status=GameStatus.SCHEDULED,
            first_pitch_utc=datetime(2024, 6, 15, 23, 10),
        )
        session.add(game)
        session.flush()

        session.add(
            PitcherStatsSnapshot(
                player_id=pitcher.id,
                as_of_date=date(2024, 6, 15),  # stats through 2024-06-14
                season=2024,
                ip=70.1,
                fip=3.62,
                xfip_proxy=None,
                k_per_9=8.9,
                bb_per_9=3.1,
                games_started=13,
                last_start_date=date(2024, 6, 10),
            )
        )
        session.add(
            OddsSnapshot(
                game_id=game.id,
                book="pinnacle",
                market=Market.H2H,
                outcome_label="New York Mets",
                decimal_odds=1.87,
                line_value=None,
                captured_at_utc=datetime(2024, 6, 15, 22, 55),
                is_closing=True,
            )
        )
        session.add(
            Pick(
                game_id=game.id,
                market=Market.H2H,
                outcome_label="New York Mets",
                line_value=None,
                decimal_odds_at_pick=1.92,
                book="pinnacle",
                model_probability=0.555,
                market_consensus_probability=0.53,
                ev=0.0656,
                stake_units=1.0,
                created_at_utc=datetime(2024, 6, 15, 18, 0),
                status=PickStatus.PENDING,
                closing_decimal_odds=None,
                clv_pct=None,
                phase=Phase.BACKTEST,
            )
        )

    with session_scope(engine) as session:
        assert session.scalars(select(Team)).all()[0].abbreviation == "NYM"
        assert session.scalars(select(Player)).one().mlb_id == 656849

        game = session.scalars(select(Game)).one()
        assert game.mlb_game_pk == 745123
        assert game.status is GameStatus.SCHEDULED
        assert game.away_starter_player_id is None

        snap = session.scalars(select(PitcherStatsSnapshot)).one()
        assert snap.as_of_date == date(2024, 6, 15)
        assert snap.fip == 3.62

        odds = session.scalars(select(OddsSnapshot)).one()
        assert odds.market is Market.H2H
        assert odds.is_closing is True
        assert odds.captured_at_utc == datetime(2024, 6, 15, 22, 55)

        pick = session.scalars(select(Pick)).one()
        assert pick.phase is Phase.BACKTEST
        assert pick.status is PickStatus.PENDING
        assert pick.clv_pct is None
