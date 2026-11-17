"""Authoritative variable and hand-designed graph definitions.

Arrows describe a probabilistic factorization, not identified causal effects.
"""

FEATURES = ["T", "D", "C", "S", "B", "H"]
VARIABLES = FEATURES + ["M"]
HAND_PARENTS = ["D", "T", "C", "H"]
HAND_PARENT_MAP = {
    "S": [],
    "D": [],
    "B": ["S"],
    "H": ["S", "B"],
    "T": ["D"],
    "C": ["D", "B"],
    "M": HAND_PARENTS,
}
HAND_EDGES = [
    (parent, node)
    for node, parents in HAND_PARENT_MAP.items()
    for parent in parents
]
