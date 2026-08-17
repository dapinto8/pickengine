"""Starting pitcher adjustment. Pure functions.

A starter's quality signal is FIP relative to league average, shrunk toward
the league average by innings pitched (simple Bayesian shrinkage:
weight = ip / (ip + SHRINKAGE_IP)), so a hot 12-inning April sample barely
moves the needle while a 150-inning season counts nearly at face value.
An unknown starter (no announced probable, or no snapshot yet — e.g. a season
debut) scores exactly league average.

The FIP gap between the two starters converts to Elo points at ELO_PER_FIP
per full run of FIP advantage. Both ELO_PER_FIP and the shrinkage prior are
tuning targets for backtesting — treat current values as sane defaults, not
truth.
"""

from typing import Protocol

LEAGUE_AVERAGE_FIP = 4.20
SHRINKAGE_IP = 40.0
ELO_PER_FIP = 40.0


class SnapshotLike(Protocol):
    """What we need from a pitcher snapshot (models.PitcherStatsSnapshot fits)."""

    fip: float
    ip: float


def shrunk_fip(fip: float, ip: float) -> float:
    """FIP shrunk toward league average by innings pitched."""
    if ip < 0:
        raise ValueError(f"innings pitched cannot be negative, got {ip}")
    weight = ip / (ip + SHRINKAGE_IP)
    return LEAGUE_AVERAGE_FIP + weight * (fip - LEAGUE_AVERAGE_FIP)


def pitcher_score(snapshot: SnapshotLike | None) -> float:
    """Shrunk FIP for a starter; league average when unknown. Lower is better."""
    if snapshot is None:
        return LEAGUE_AVERAGE_FIP
    return shrunk_fip(snapshot.fip, snapshot.ip)


def pitching_adjustment_elo(
    home_score: float, away_score: float, elo_per_fip: float = ELO_PER_FIP
) -> float:
    """Elo adjustment from the home team's perspective.

    Scores are shrunk FIPs (lower = better), so a home starter with FIP 1.00
    lower than the away starter is worth +elo_per_fip to the home side.
    """
    return (away_score - home_score) * elo_per_fip
