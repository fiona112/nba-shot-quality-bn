
from itertools import combinations, permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from src.bayesian_network import HandDesignedBN
from src.chronological_evaluation import load_all_data


TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")

N_FOLDS = 5
ALPHA = 1.0

# Players need enough attempts for FG% to be meaningful.
MIN_SHOTS = 200
MIN_SHOTS_PER_HALF = 100
LABELS_PER_SIDE = 5


def cross_fit_expected(df):
    """
    Give every shot an out-of-fold expected make probability.

    Folds are split by game, so a shot's CPT is never
    estimated from the game it came from.
    """
    expected = np.empty(len(df))
    folds = GroupKFold(n_splits=N_FOLDS)

    for fit_idx, pred_idx in folds.split(df, groups=df["GAME_ID"]):
        model = HandDesignedBN(alpha=ALPHA)
        model.fit(df.iloc[fit_idx])
        expected[pred_idx] = model.predict_proba(df.iloc[pred_idx])

    return expected


def summarize_players(df):
    """
    Compare each player's actual FG% with the FG% the
    model expects from the shots they took.
    """
    df = df.assign(
        variance=df["expected"] * (1 - df["expected"])
    )

    players = (
        df.groupby("player_name")
        .agg(
            shots=("M", "size"),
            makes=("M", "sum"),
            expected_makes=("expected", "sum"),
            variance=("variance", "sum"),
        )
        .reset_index()
    )

    players["actual_fg"] = players["makes"] / players["shots"]
    players["expected_fg"] = players["expected_makes"] / players["shots"]

    # FG% over expectation: shot-making beyond shot selection.
    players["fg_over_expected"] = (
        players["actual_fg"] - players["expected_fg"]
    )

    # If shots were independent Bernoulli(expected) draws,
    # this is how many standard errors the gap is from zero.
    players["z_score"] = (
        (players["makes"] - players["expected_makes"])
        / np.sqrt(players["variance"])
    )

    players["player_name"] = players["player_name"].str.title()

    return players.drop(columns="variance")


def split_half_stability(df):
    """
    Correlate FG% over expectation between the two halves
    of the season. A stable gap points to shooter skill;
    a gap near zero correlation points to noise.
    """
    midpoint = df["date"].sort_values().iloc[len(df) // 2]

    halves = []
    for name, half in [
        ("first", df[df["date"] < midpoint]),
        ("second", df[df["date"] >= midpoint]),
    ]:
        players = (
            half.groupby("player_name")
            .agg(shots=("M", "size"), makes=("M", "sum"),
                 expected_makes=("expected", "sum"))
        )
        players[name] = (
            (players["makes"] - players["expected_makes"])
            / players["shots"]
        )
        halves.append(
            players.loc[players["shots"] >= MIN_SHOTS_PER_HALF, [name]]
        )

    paired = halves[0].join(halves[1], how="inner")
    correlation = paired["first"].corr(paired["second"])

    return correlation, len(paired), midpoint


def segments_cross(a, b, c, d):
    """True if segment a-b properly intersects segment c-d."""
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1])
                       - (q[1] - p[1]) * (r[0] - p[0]))
    return (side(a, b, c) * side(a, b, d) < 0
            and side(c, d, a) * side(c, d, b) < 0)


def order_labels(rows, text_x, text_ys):
    """Assign rows to label slots so the fewest leader lines cross."""
    points = list(zip(rows["expected_fg"], rows["actual_fg"]))
    best_order, best_crossings = None, None

    for order in permutations(range(len(points))):
        lines = [((text_x, y), points[i]) for y, i in zip(text_ys, order)]
        crossings = sum(
            segments_cross(*lines[i], *lines[j])
            for i, j in combinations(range(len(lines)), 2)
        )
        if best_crossings is None or crossings < best_crossings:
            best_order, best_crossings = order, crossings

    return rows.iloc[list(best_order)]


def plot_players(players):
    qualified = players[players["shots"] >= MIN_SHOTS]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        qualified["expected_fg"],
        qualified["actual_fg"],
        s=qualified["shots"] / 20,
        alpha=0.5
    )

    low = min(qualified["expected_fg"].min(), qualified["actual_fg"].min())
    high = max(qualified["expected_fg"].max(), qualified["actual_fg"].max())
    ax.plot([low, high], [low, high], linestyle="--", color="gray",
            label="Actual = expected")

    # Stack the labels in two columns (above the diagonal at top-left,
    # below it at bottom-right) with leader lines, so none overlap.
    ranked = qualified.sort_values("fg_over_expected")
    span = high - low
    groups = [
        (ranked.tail(LABELS_PER_SIDE).sort_values("actual_fg", ascending=False),
         low, high, "left"),
        (ranked.head(LABELS_PER_SIDE).sort_values("actual_fg", ascending=False),
         high, low + 0.25 * span, "right"),
    ]
    for players_to_label, text_x, top_y, align in groups:
        text_ys = [top_y - i * 0.035 * span
                   for i in range(len(players_to_label))]
        players_to_label = order_labels(players_to_label, text_x, text_ys)
        for text_y, (_, row) in zip(text_ys, players_to_label.iterrows()):
            ax.annotate(
                row["player_name"],
                (row["expected_fg"], row["actual_fg"]),
                xytext=(text_x, text_y),
                fontsize=8,
                ha=align,
                va="center",
                arrowprops=dict(
                    arrowstyle="-", color="gray", lw=0.6,
                    # Start the line at the name's inner edge.
                    relpos=(1, 0.5) if align == "left" else (0, 0.5)
                )
            )

    ax.set_xlabel("Expected FG% (hand-designed BN, out-of-fold)")
    ax.set_ylabel("Actual FG%")
    ax.set_title(
        f"Actual vs. expected FG% (players with {MIN_SHOTS}+ shots)"
    )
    ax.legend(loc="upper center")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "player_actual_vs_expected.png", dpi=200)
    plt.close(fig)


def main():
    df = load_all_data().reset_index(drop=True)
    df["expected"] = cross_fit_expected(df)

    print("=== OVERALL ===")
    print("Shots:", len(df))
    print("Actual FG%:", round(df["M"].mean(), 4))
    print("Mean expected FG%:", round(df["expected"].mean(), 4))

    players = summarize_players(df)
    qualified = players[players["shots"] >= MIN_SHOTS]

    print(f"\nPlayers with {MIN_SHOTS}+ shots:", len(qualified))
    print(
        "Spread (std) of expected FG%:",
        round(qualified["expected_fg"].std(), 4)
    )
    print(
        "Spread (std) of FG% over expected:",
        round(qualified["fg_over_expected"].std(), 4)
    )
    print(
        "Players more than 2 SE from expectation:",
        (qualified["z_score"].abs() > 2).sum()
    )

    columns = ["player_name", "shots", "actual_fg",
               "expected_fg", "fg_over_expected", "z_score"]
    ranked = qualified.sort_values("fg_over_expected", ascending=False)
    print("\nMost above expectation:")
    print(ranked[columns].head(10).to_string(index=False))
    print("\nMost below expectation:")
    print(ranked[columns].tail(10).to_string(index=False))

    correlation, n_players, midpoint = split_half_stability(df)
    print("\n=== SPLIT-HALF STABILITY ===")
    print("Season split at:", midpoint.date())
    print(f"Players with {MIN_SHOTS_PER_HALF}+ shots in both halves:",
          n_players)
    print("Correlation of FG% over expected between halves:",
          round(correlation, 3))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    players.sort_values("fg_over_expected", ascending=False).to_csv(
        TABLE_DIR / "player_shot_quality.csv",
        index=False
    )
    pd.DataFrame([{
        "split_date": midpoint.date(),
        "min_shots_per_half": MIN_SHOTS_PER_HALF,
        "players": n_players,
        "correlation": correlation
    }]).to_csv(
        TABLE_DIR / "player_split_half_stability.csv",
        index=False
    )
    plot_players(players)

    print("\nSaved files:")
    print(TABLE_DIR / "player_shot_quality.csv")
    print(TABLE_DIR / "player_split_half_stability.csv")
    print(FIG_DIR / "player_actual_vs_expected.png")


if __name__ == "__main__":
    main()
