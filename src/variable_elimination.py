
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")

# Parent sets from our proposed BN
PARENTS = {
    "S": [],
    "D": [],
    "B": ["S"],
    "H": ["S", "B"],
    "T": ["D"],
    "C": ["D", "B"],
    "M": ["D", "T", "C", "H"],
}

VARIABLES = list(PARENTS.keys())


class Factor:
    """
    A probability factor stored as a pandas DataFrame.

    Example:
        D     C      p
        <5    0-2    0.3
        <5    2-4    0.7
    """

    def __init__(self, table):
        self.table = table.copy()

    @property
    def variables(self):
        return [c for c in self.table.columns if c != "p"]

    def restrict(self, variable, value):
        """Condition on observed evidence."""
        if variable not in self.variables:
            return Factor(self.table)

        result = self.table[
            self.table[variable] == str(value)
        ].copy()

        result = result.drop(columns=[variable])
        return Factor(result.reset_index(drop=True))

    def multiply(self, other):
        """Multiply two factors over shared variables."""
        shared = [
            v for v in self.variables
            if v in other.variables
        ]

        if shared:
            merged = self.table.merge(
                other.table,
                on=shared,
                how="inner",
                suffixes=("_left", "_right")
            )
        else:
            merged = self.table.merge(
                other.table,
                how="cross",
                suffixes=("_left", "_right")
            )

        merged["p"] = (
            merged["p_left"] * merged["p_right"]
        )

        return Factor(
            merged.drop(columns=["p_left", "p_right"])
        )

    def sum_out(self, variable):
        """Marginalize a variable out of this factor."""
        if variable not in self.variables:
            return Factor(self.table)

        remaining = [
            v for v in self.variables
            if v != variable
        ]

        if not remaining:
            return Factor(
                pd.DataFrame({
                    "p": [self.table["p"].sum()]
                })
            )

        result = (
            self.table
            .groupby(remaining, as_index=False)["p"]
            .sum()
        )

        return Factor(result)

    def normalize(self):
        """Normalize factor probabilities to sum to 1."""
        result = self.table.copy()
        total = result["p"].sum()

        if total <= 0:
            raise ValueError(
                "Cannot normalize a zero-probability factor."
            )

        result["p"] /= total
        return Factor(result)


class BayesianNetwork:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.states = {}
        self.factors = {}

    def fit(self, data):
        df = data.copy()

        # Convert every discrete variable to string states,
        # preserving missing values.
        for variable in VARIABLES:
            df[variable] = df[variable].astype("string")

        self.states = {
            variable: sorted(
                df[variable].dropna().unique().tolist()
            )
            for variable in VARIABLES
        }

        # Estimate each P(node | parents) separately.
        for node, parents in PARENTS.items():
            columns = parents + [node]

            # Missing values are excluded only for the
            # variables needed by this particular CPT.
            subset = df[columns].dropna()

            joint_counts = Counter(
                tuple(row)
                for row in subset.itertuples(
                    index=False, name=None
                )
            )

            parent_counts = Counter(
                tuple(row)
                for row in subset[parents].itertuples(
                    index=False, name=None
                )
            ) if parents else Counter({
                (): len(subset)
            })

            configurations = product(
                *[self.states[p] for p in parents]
            )

            rows = []
            k = len(self.states[node])

            for parent_values in configurations:
                parent_values = tuple(parent_values)
                total = parent_counts[parent_values]

                for node_value in self.states[node]:
                    count = joint_counts[
                        parent_values + (node_value,)
                    ]

                    probability = (
                        (count + self.alpha) /
                        (total + self.alpha * k)
                    )

                    row = dict(zip(parents, parent_values))
                    row[node] = node_value
                    row["p"] = probability
                    rows.append(row)

            self.factors[node] = Factor(
                pd.DataFrame(rows, columns=columns + ["p"])
            )

        return self

    def query(self, target="M", evidence=None):
        """
        Compute P(target | evidence) using
        variable elimination.
        """
        evidence = evidence or {}

        if target in evidence:
            raise ValueError(
                "Target cannot also be in evidence."
            )

        for variable, value in evidence.items():
            if variable not in self.states:
                raise ValueError(
                    f"Unknown variable: {variable}"
                )
            if str(value) not in self.states[variable]:
                raise ValueError(
                    f"Unknown state {value} for {variable}"
                )

        factors = [
            Factor(f.table)
            for f in self.factors.values()
        ]

        # Restrict every relevant factor to evidence.
        for variable, value in evidence.items():
            factors = [
                f.restrict(variable, value)
                for f in factors
            ]

        # Fixed elimination order.
        # We will optimize this order later if needed.
        elimination_order = [
            "S", "B", "D", "T", "C", "H", "M"
        ]

        for variable in elimination_order:
            if variable == target or variable in evidence:
                continue

            involved = [
                f for f in factors
                if variable in f.variables
            ]

            if not involved:
                continue

            factors = [
                f for f in factors
                if variable not in f.variables
            ]

            combined = involved[0]
            for factor in involved[1:]:
                combined = combined.multiply(factor)

            factors.append(combined.sum_out(variable))

        # Multiply the remaining factors.
        result = factors[0]
        for factor in factors[1:]:
            result = result.multiply(factor)

        result = result.normalize()

        return {
            row[target]: float(row["p"])
            for _, row in result.table.iterrows()
        }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    model = BayesianNetwork(alpha=1.0)
    model.fit(train)

    print("=== LEARNED CPT SIZES ===")
    for node, factor in model.factors.items():
        print(
            f"P({node} | {PARENTS[node]}): "
            f"{len(factor.table)} rows"
        )

    print("\n=== QUERY 1: NO EVIDENCE ===")
    print(model.query("M"))

    print("\n=== QUERY 2: PARTIAL EVIDENCE ===")
    print(model.query("M", {
        "T": "3",
        "D": "30+",
        "H": "<2"
    }))

    print("\n=== QUERY 3: FULL EVIDENCE ===")
    example = test.dropna(
        subset=["T", "D", "C", "S", "B", "H"]
    ).iloc[0]

    evidence = {
        variable: str(example[variable])
        for variable in ["T", "D", "C", "S", "B", "H"]
    }

    print("Evidence:", evidence)
    print("Prediction:", model.query("M", evidence))

    print("\n=== BASIC VALIDATION ===")
    queries = [
        {},
        {"T": "3"},
        {"T": "3", "D": "30+"},
        {"T": "3", "D": "30+", "H": "<2"},
        evidence
    ]

    for i, e in enumerate(queries, start=1):
        probabilities = model.query("M", e)

        assert np.isclose(
            sum(probabilities.values()), 1.0,
            atol=1e-10
        )

        assert all(
            0 <= p <= 1
            for p in probabilities.values()
        )

        print(f"Query {i}: PASS")

    print("\nVariable elimination is working!")


if __name__ == "__main__":
    main()
