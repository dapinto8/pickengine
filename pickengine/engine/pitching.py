"""Starting pitcher adjustments.

Pure functions that adjust the Elo-based team win probability for the
announced starting pitchers (e.g. a pitcher quality rating relative to the
team's rotation average, built from pre-game stats only).

Uses only pitcher information available before first pitch: the announced
starter and stats through the pitcher's previous appearance. If no starter
was announced pre-game, the adjustment is neutral — never backfilled from
the actual starter.
"""
