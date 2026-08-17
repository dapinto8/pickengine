"""Elo math and replay tests against hand-computed examples."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from pickengine.engine.elo import (
    GAME_COMPLETION_LAG,
    INITIAL_RATING,
    EloRatings,
    elo_expectation,
    rating_delta,
    regress_toward_mean,
)


@dataclass
class FakeGame:
    home_team_id: int
    away_team_id: int
    season: int
    first_pitch_utc: datetime | None
    home_score: int | None
    away_score: int | None


def game(
    home: int, away: int, when: datetime, home_score: int, away_score: int, season: int = 2024
) -> FakeGame:
    return FakeGame(home, away, season, when, home_score, away_score)


# Expected home score for two equal 1500s with +24 home advantage.
P_HOME_EQUAL = 1 / (1 + 10 ** (-24 / 400))  # ~0.534483
DELTA_EQUAL_HOME_WIN = 4 * (1 - P_HOME_EQUAL)  # ~1.862068


def test_expectation_hand_computed() -> None:
    assert elo_expectation(1500, 1500) == 0.5
    assert elo_expectation(1600, 1500) == pytest.approx(0.6400649998, abs=1e-9)
    assert elo_expectation(1500, 1600) == pytest.approx(1 - 0.6400649998, abs=1e-9)


def test_rating_delta() -> None:
    assert rating_delta(expected=0.5, actual=1.0) == pytest.approx(2.0)  # K=4
    assert rating_delta(expected=0.75, actual=0.0) == pytest.approx(-3.0)


def test_regress_toward_mean() -> None:
    assert regress_toward_mean(1560.0) == pytest.approx(1540.0)
    assert regress_toward_mean(1440.0) == pytest.approx(1460.0)
    assert regress_toward_mean(1500.0) == 1500.0


def test_single_game_update_hand_computed() -> None:
    when = datetime(2024, 6, 1, 19, 0)
    elo = EloRatings([game(1, 2, when, home_score=5, away_score=3)])
    after = when + GAME_COMPLETION_LAG + timedelta(seconds=1)
    assert elo.get_rating(1, after) == pytest.approx(1500 + DELTA_EQUAL_HOME_WIN)
    assert elo.get_rating(2, after) == pytest.approx(1500 - DELTA_EQUAL_HOME_WIN)


def test_get_rating_ignores_games_on_or_after_as_of() -> None:
    when = datetime(2024, 6, 1, 19, 0)
    elo = EloRatings([game(1, 2, when, 5, 3)])
    # At first pitch, during the game, and at the exact availability instant
    # the result is unknown; only strictly after does it count.
    assert elo.get_rating(1, when) == INITIAL_RATING
    assert elo.get_rating(1, when + timedelta(hours=2)) == INITIAL_RATING
    assert elo.get_rating(1, when + GAME_COMPLETION_LAG) == INITIAL_RATING
    assert elo.get_rating(1, when + GAME_COMPLETION_LAG + timedelta(seconds=1)) > INITIAL_RATING
    assert elo.get_rating(999, datetime(2030, 1, 1)) == INITIAL_RATING  # unknown team


def test_replay_is_chronological_not_insertion_order() -> None:
    early = datetime(2024, 6, 1, 19, 0)
    late = datetime(2024, 6, 8, 19, 0)
    g_early = game(1, 2, early, 5, 3)
    g_late = game(1, 3, late, 4, 2)
    # Same games, opposite insertion orders -> identical timelines.
    a = EloRatings([g_early, g_late])
    b = EloRatings([g_late, g_early])
    probe = late + GAME_COMPLETION_LAG + timedelta(seconds=1)
    assert a.get_rating(1, probe) == b.get_rating(1, probe)
    assert a.get_rating(1, probe) != INITIAL_RATING


def test_adding_an_earlier_game_changes_subsequent_ratings() -> None:
    early = datetime(2024, 6, 1, 19, 0)
    late = datetime(2024, 6, 8, 19, 0)
    g_late = game(1, 3, late, 4, 2)
    probe = late + GAME_COMPLETION_LAG + timedelta(seconds=1)

    without_early = EloRatings([g_late])
    with_early = EloRatings([g_late, game(1, 2, early, 5, 3)])

    # Team 1 entered the late game already above 1500, so its post-game
    # rating must differ (higher) once the earlier win is included.
    assert with_early.get_rating(1, probe) > without_early.get_rating(1, probe)
    # And between the games only the early result is visible.
    mid = early + GAME_COMPLETION_LAG + timedelta(seconds=1)
    assert with_early.get_rating(1, mid) == pytest.approx(1500 + DELTA_EQUAL_HOME_WIN)


def test_season_boundary_regression() -> None:
    g2023 = game(1, 2, datetime(2023, 6, 1, 19, 0), 5, 3, season=2023)
    g2024 = game(1, 2, datetime(2024, 6, 1, 19, 0), 5, 3, season=2024)
    elo = EloRatings([g2023, g2024])

    # After the 2023 game, before the boundary.
    end_2023 = 1500 + DELTA_EQUAL_HOME_WIN
    assert elo.get_rating(1, datetime(2023, 12, 31)) == pytest.approx(end_2023)
    # In 2024 before any games: regressed 1/3 toward 1500.
    regressed = 1500 + DELTA_EQUAL_HOME_WIN * (2 / 3)
    assert elo.get_rating(1, datetime(2024, 3, 1)) == pytest.approx(regressed)
    assert elo.get_rating(2, datetime(2024, 3, 1)) == pytest.approx(
        1500 - DELTA_EQUAL_HOME_WIN * (2 / 3)
    )


def test_multi_season_gap_applies_regression_per_season() -> None:
    g2023 = game(1, 2, datetime(2023, 6, 1, 19, 0), 5, 3, season=2023)
    g2025 = game(3, 4, datetime(2025, 6, 1, 19, 0), 5, 3, season=2025)
    elo = EloRatings([g2023, g2025])
    regressed_twice = 1500 + DELTA_EQUAL_HOME_WIN * (2 / 3) ** 2
    assert elo.get_rating(1, datetime(2025, 3, 1)) == pytest.approx(regressed_twice)


def test_incomplete_game_raises() -> None:
    with pytest.raises(ValueError, match="final games"):
        EloRatings([game(1, 2, datetime(2024, 6, 1), 5, None)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="final games"):
        EloRatings([FakeGame(1, 2, 2024, None, 5, 3)])
