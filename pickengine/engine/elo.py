"""Elo-style team ratings. Pure computation — callers supply the games.

Classic Elo tuned for MLB: initial 1500, K=4 (baseball outcomes are noisy, so
low K), home advantage of +24 Elo applied at expectation time — both when
predicting and when computing the update expectation — but NEVER baked into
stored ratings. At each season boundary every team regresses 1/3 of the way
back to 1500 (effective January 1 of the new season).

Lookahead discipline: a game's result becomes available at first_pitch_utc +
GAME_COMPLETION_LAG (we do not store completion times, so a conservative
fixed lag stands in for game duration). `get_rating(team_id, as_of)` returns
the rating using only games whose availability time is strictly before as_of.
Rating timelines are precomputed per team in the constructor, so lookups are
O(log n).
"""

from bisect import bisect_left
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Protocol

INITIAL_RATING = 1500.0
K_FACTOR = 4.0
HOME_ADVANTAGE_ELO = 24.0
SEASON_REGRESSION = 1 / 3
GAME_COMPLETION_LAG = timedelta(hours=4)


class FinalGame(Protocol):
    """What EloRatings needs from a game row (models.Game satisfies this)."""

    home_team_id: int
    away_team_id: int
    season: int
    first_pitch_utc: datetime | None
    home_score: int | None
    away_score: int | None


def elo_expectation(rating_a: float, rating_b: float) -> float:
    """Expected score of A against B (0..1)."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def rating_delta(expected: float, actual: float, k: float = K_FACTOR) -> float:
    """Rating change for the side whose expectation was `expected`."""
    return k * (actual - expected)


def regress_toward_mean(
    rating: float, fraction: float = SEASON_REGRESSION, mean: float = INITIAL_RATING
) -> float:
    return rating + (mean - rating) * fraction


class EloRatings:
    """Rating timelines built by replaying final games chronologically.

    Games are sorted by first pitch internally, so insertion order does not
    matter. Every game must have first_pitch_utc and both scores — pass only
    FINAL games; anything incomplete raises. `k` overrides K_FACTOR (tuning).
    """

    def __init__(self, games: Iterable[FinalGame], k: float = K_FACTOR) -> None:
        self.k = k
        self._times: dict[int, list[datetime]] = {}
        self._ratings: dict[int, list[float]] = {}

        ordered = sorted(games, key=lambda g: g.first_pitch_utc or datetime.min)
        current: dict[int, float] = {}
        current_season: int | None = None
        for game in ordered:
            if game.first_pitch_utc is None or game.home_score is None or game.away_score is None:
                raise ValueError(
                    "EloRatings requires final games with first_pitch_utc and scores; "
                    f"got incomplete game for teams {game.home_team_id} vs {game.away_team_id}"
                )
            if current_season is None:
                current_season = game.season
            while game.season > current_season:
                current_season += 1
                boundary = datetime(current_season, 1, 1)
                for team_id in current:
                    current[team_id] = regress_toward_mean(current[team_id])
                    self._append(team_id, boundary, current[team_id])

            home = current.get(game.home_team_id, INITIAL_RATING)
            away = current.get(game.away_team_id, INITIAL_RATING)
            expected_home = elo_expectation(home + HOME_ADVANTAGE_ELO, away)
            if game.home_score > game.away_score:
                actual_home = 1.0
            elif game.home_score < game.away_score:
                actual_home = 0.0
            else:
                actual_home = 0.5
            delta = rating_delta(expected_home, actual_home, k=self.k)

            available_at = game.first_pitch_utc + GAME_COMPLETION_LAG
            current[game.home_team_id] = home + delta
            current[game.away_team_id] = away - delta
            self._append(game.home_team_id, available_at, home + delta)
            self._append(game.away_team_id, available_at, away - delta)

    def _append(self, team_id: int, available_at: datetime, rating: float) -> None:
        self._times.setdefault(team_id, []).append(available_at)
        self._ratings.setdefault(team_id, []).append(rating)

    def get_rating(self, team_id: int, as_of: datetime) -> float:
        """Rating from games available strictly before as_of (else 1500)."""
        times = self._times.get(team_id)
        if not times:
            return INITIAL_RATING
        idx = bisect_left(times, as_of)
        return self._ratings[team_id][idx - 1] if idx > 0 else INITIAL_RATING

    def last_update_time(self, team_id: int, as_of: datetime) -> datetime | None:
        """Availability time of the newest update get_rating(as_of) sees.

        Exists so integrity checks can assert the rating truly predates the
        moment it is used.
        """
        times = self._times.get(team_id)
        if not times:
            return None
        idx = bisect_left(times, as_of)
        return times[idx - 1] if idx > 0 else None
