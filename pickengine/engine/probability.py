"""Final probability: combines the model layers into one probability per side.

Pure functions that blend:
1. The de-vigged market consensus probability (from devig.py).
2. The model probability (Elo from elo.py, adjusted by pitching.py).

into the final probability `p` used for EV screening. The blend weight is an
explicit parameter — the market baseline is the anchor and the model moves
probability away from it only as far as its demonstrated edge justifies.

Outputs are floats in (0, 1); complementary sides must sum to 1.
"""
