"""Exploratory, game-cross-fitted player residual comparisons.

Residuals combine shooting, omitted context, sampling noise, and model error;
they are not identified player-skill effects. This analysis pools the verified
cleaned dataset and is separate from the five-model holdout evaluation.
"""

from itertools import combinations, permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import GroupKFold

from src.bayesian_network import HandDesignedBN, MAKE_PARENTS
from src.chronological_evaluation import load_all_data
from src.final_evaluation import load_final_data
from src.validation import write_validation_metadata


TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
DATA_DIR = Path("data/processed")

N_FOLDS = 5
ALPHA = 1.0

# Players need enough attempts for FG% to be meaningful.
MIN_SHOTS = 200
MIN_SHOTS_PER_HALF = 100
LABELS_PER_SIDE = 5


def cross_fit_predictions(df, n_folds=N_FOLDS):
    """Return identifiable OOF predictions and parent support for every shot.

    The hand graph and alpha are fixed; no outcome-based fold or model choice
    occurs. Other games from the same player can enter a fold's training set.
    """
    required = ["RAW_ROW_ID", "GAME_ID", "player_id", "player_name", "date", *MAKE_PARENTS, "M"]
    if not set(required).issubset(df.columns) or df[required].isna().any().any():
        raise ValueError("Cross-fitting requires complete shot/game/player identifiers, dates, parents, and targets.")
    if not df["RAW_ROW_ID"].is_unique or not df["M"].isin([0, 1]).all():
        raise ValueError("Cross-fitting requires unique shot identifiers and binary targets.")
    if n_folds < 2 or df["GAME_ID"].nunique() < n_folds:
        raise ValueError("Cross-fitting needs at least two folds and one game per fold.")
    df = df.reset_index(drop=True)
    expected = np.full(len(df), np.nan)
    support = np.zeros(len(df), dtype=int)
    assignments = np.full(len(df), -1, dtype=int)
    fold_rows = []
    folds = GroupKFold(n_splits=n_folds)

    for fold, (fit_idx, pred_idx) in enumerate(folds.split(df, groups=df["GAME_ID"])):
        fitting, prediction = df.iloc[fit_idx], df.iloc[pred_idx]
        if not set(fitting["GAME_ID"]).isdisjoint(prediction["GAME_ID"]):
            raise ValueError("Cross-fitting leaked a prediction game into training.")
        model = HandDesignedBN(alpha=ALPHA)
        model.fit(fitting)
        expected[pred_idx] = model.predict_proba(prediction)
        matched = prediction[MAKE_PARENTS].merge(
            model.cpt[[*MAKE_PARENTS, "count"]], on=MAKE_PARENTS,
            how="left", sort=False, validate="many_to_one",
        )
        support[pred_idx] = matched["count"].fillna(0).to_numpy(dtype=int)
        assignments[pred_idx] = fold
        fold_rows.append({
            "fold": fold, "training_shots": len(fitting), "prediction_shots": len(prediction),
            "training_games": int(fitting["GAME_ID"].nunique()),
            "prediction_games": int(prediction["GAME_ID"].nunique()),
            "unseen_parent_shots": int((support[pred_idx] == 0).sum()),
            "minimum_training_parent_count": int(support[pred_idx].min()),
        })
    if not np.isfinite(expected).all() or (assignments < 0).any() or not ((expected > 0) & (expected < 1)).all():
        raise ValueError("Cross-fitting did not produce one valid smoothed probability per shot.")
    predictions = df[["RAW_ROW_ID", "GAME_ID", "player_id", "player_name", "date", "M"]].copy()
    predictions["fold"] = assignments
    predictions["expected"] = expected
    predictions["training_parent_count"] = support
    return predictions, pd.DataFrame(fold_rows)


def cross_fit_expected(df):
    """Compatibility wrapper returning only game-cross-fitted probabilities."""
    return cross_fit_predictions(df)[0]["expected"].to_numpy()


def summarize_players(df):
    """
    Compare actual rates with a fixed model's OOF expected rates.

    The Bernoulli reference z ignores game/player dependence and estimation
    uncertainty. It is descriptive and is not a calibrated significance test.
    """
    df = df.assign(
        variance=df["expected"] * (1 - df["expected"])
    )

    players = (
        df.groupby(["player_id", "player_name"])
        .agg(
            shots=("M", "size"),
            games=("GAME_ID", "nunique"),
            makes=("M", "sum"),
            expected_makes=("expected", "sum"),
            variance=("variance", "sum"),
            unseen_parent_shots=("training_parent_count", lambda values: int(values.eq(0).sum())),
            minimum_training_parent_count=("training_parent_count", "min"),
        )
        .reset_index()
    )

    players["actual_fg"] = players["makes"] / players["shots"]
    players["expected_fg"] = players["expected_makes"] / players["shots"]

    # Residual after accounting for the graph's measured binned predictors.
    players["fg_over_expected"] = (
        players["actual_fg"] - players["expected_fg"]
    )

    # Reference scale only: expected probabilities are estimated and games
    # share players, so this is not a calibrated standard-error calculation.
    players["bernoulli_reference_z"] = (
        (players["makes"] - players["expected_makes"])
        / np.sqrt(players["variance"])
    )

    players["player_name"] = players["player_name"].str.title()

    return players.drop(columns="variance")


def cross_fitted_split_halves(df, n_folds=N_FOLDS):
    """Compare residuals with separate within-half game cross-fitting.

    No first-half prediction uses second-half outcomes, or vice versa. The
    correlation is descriptive; persistent omitted context can also persist.
    """
    midpoint = df["date"].sort_values().iloc[len(df) // 2]
    halves, prediction_tables, fold_tables = [], [], []
    cohort_counts = {}
    for name, half in [
        ("first", df[df["date"] < midpoint]),
        ("second", df[df["date"] >= midpoint]),
    ]:
        predictions, folds = cross_fit_predictions(half, n_folds=n_folds)
        prediction_tables.append(predictions.assign(half=name))
        fold_tables.append(folds.assign(analysis=name))
        cohort_counts.update({
            f"{name}_shots": len(half), f"{name}_games": int(half["GAME_ID"].nunique()),
        })
        players = (
            predictions.groupby("player_id")
            .agg(shots=("M", "size"), makes=("M", "sum"),
                 expected_makes=("expected", "sum"))
        )
        players[name] = (
            (players["makes"] - players["expected_makes"])
            / players["shots"]
        )
        halves.append(
            players.loc[players["shots"] >= MIN_SHOTS_PER_HALF, [name, "shots"]]
            .rename(columns={"shots": f"{name}_shots"})
        )

    paired = halves[0].join(halves[1], how="inner")
    correlation = paired["first"].corr(paired["second"]) if len(paired) > 1 else np.nan
    summary = pd.DataFrame([{
        "split_date": str(midpoint.date()), "min_shots_per_half": MIN_SHOTS_PER_HALF,
        "players": len(paired), "correlation": correlation, **cohort_counts,
    }])
    return summary, paired.reset_index(), pd.concat(prediction_tables, ignore_index=True), pd.concat(fold_tables, ignore_index=True)


def split_half_stability(df):
    """Compatibility wrapper around independently cross-fitted halves."""
    summary, _, _, _ = cross_fitted_split_halves(df)
    row = summary.iloc[0]
    return row["correlation"], int(row["players"]), pd.Timestamp(row["split_date"])


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
    if qualified.empty:
        raise ValueError("No players meet the prespecified plotting support threshold.")

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
    span = max(high - low, 0.05)
    groups = [
        (ranked.tail(LABELS_PER_SIDE).sort_values("actual_fg", ascending=False),
         low, high, "left"),
        (ranked.head(LABELS_PER_SIDE).sort_values("actual_fg", ascending=False),
         high, low + 0.3 * span, "right"),
    ]
    for players_to_label, text_x, top_y, align in groups:
        text_ys = [top_y - i * 0.075 * span
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
        f"Game-cross-fitted player residuals ({MIN_SHOTS}+ shots)"
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center")
    fig.text(0.5, 0.01, "Residuals also reflect omitted context, sampling noise, and model error.",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "player_actual_vs_expected.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    train, test, cohort = load_final_data(DATA_DIR)
    df = load_all_data(train, test).reset_index(drop=True)
    if df[["player_id", "player_name"]].isna().any().any() or not df["RAW_ROW_ID"].is_unique:
        raise ValueError("Player comparison requires known player identifiers/names and unique shots.")
    if df.groupby("player_id")["player_name"].nunique().gt(1).any():
        raise ValueError("Player identifiers map to inconsistent names; resolve aliases explicitly.")
    predictions, full_folds = cross_fit_predictions(df)
    df["expected"] = predictions["expected"].to_numpy()
    df["training_parent_count"] = predictions["training_parent_count"].to_numpy()

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
    print("Unseen OOF parent combinations:", int(predictions["training_parent_count"].eq(0).sum()))

    columns = ["player_name", "shots", "actual_fg",
               "expected_fg", "fg_over_expected", "bernoulli_reference_z"]
    ranked = qualified.sort_values("fg_over_expected", ascending=False)
    print("\nMost above expectation:")
    print(ranked[columns].head(10).to_string(index=False))
    print("\nMost below expectation:")
    print(ranked[columns].tail(10).to_string(index=False))

    stability, paired, half_predictions, half_folds = cross_fitted_split_halves(df)
    stability_row = stability.iloc[0]
    print("\n=== SPLIT-HALF STABILITY ===")
    print("Date split at:", stability_row["split_date"])
    print(f"Players with {MIN_SHOTS_PER_HALF}+ shots in both halves:",
          int(stability_row["players"]))
    print("Correlation of FG% over expected between halves:",
          round(stability_row["correlation"], 3))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    players.sort_values("fg_over_expected", ascending=False).to_csv(
        TABLE_DIR / "player_shot_quality.csv",
        index=False
    )
    stability.to_csv(
        TABLE_DIR / "player_split_half_stability.csv",
        index=False
    )
    paired.to_csv(TABLE_DIR / "player_split_half_pairs.csv", index=False)
    pd.concat([full_folds.assign(analysis="full"), half_folds], ignore_index=True).to_csv(
        TABLE_DIR / "player_crossfit_folds.csv", index=False,
    )
    # Keep detailed shot-level artifacts available locally without large PR diffs.
    predictions.to_csv(DATA_DIR / "player_crossfit_predictions.csv", index=False)
    half_predictions.to_csv(DATA_DIR / "player_split_half_predictions.csv", index=False)
    plot_players(players)
    outputs = [
        TABLE_DIR / "player_shot_quality.csv", TABLE_DIR / "player_split_half_stability.csv",
        TABLE_DIR / "player_split_half_pairs.csv", TABLE_DIR / "player_crossfit_folds.csv",
        FIG_DIR / "player_actual_vs_expected.png", DATA_DIR / "player_crossfit_predictions.csv",
        DATA_DIR / "player_split_half_predictions.csv",
    ]
    write_validation_metadata(
        TABLE_DIR / "player_comparison_metadata.json", "exploratory_player_comparison", cohort,
        {
            "data_used": "complete hash-verified cleaned dataset; exploratory analysis, separate from holdout evaluation",
            "shots": len(df), "games": int(df["GAME_ID"].nunique()),
            "model": "fixed hand-designed make CPT", "parents": MAKE_PARENTS,
            "alpha": ALPHA, "folds": N_FOLDS,
            "fold_method": "deterministic sklearn GroupKFold by GAME_ID; no game in its own prediction training set",
            "player_grouping": "player_id with checked one-to-one player names",
            "min_shots": MIN_SHOTS, "min_shots_per_half": MIN_SHOTS_PER_HALF,
            "split_half_method": "median shot date, complete dates; separate game cross-fitting inside each half",
            "split_date": stability_row["split_date"],
            "limitations": [
                "Player residuals combine shooting, omitted context, coarse feature bins, sampling noise, and model error; they do not identify player skill or causal effects.",
                "All cleaned shots, including the earlier random test cohort, enter this descriptive cross-fit. It is not new holdout validation or a model-selection result.",
                "Other games from the same players may occur in fold training. This is game cross-fitting rather than evaluation on unseen players.",
                "Bernoulli reference z statistics condition on estimated probabilities, assume independent shots, ignore fitting/game/player dependence, and are not calibrated significance tests. No significance or multiple-comparison claim is made.",
                "Split-half correlation uses only players with sufficient shots in both halves. It is descriptive and can reflect persistent omitted context; neither its presence nor absence establishes skill or noise alone.",
                "The dataset covers a partial 2014-15 season; the date split is not a full-season skill reliability estimate.",
            ],
        }, outputs,
    )

    print("\nSaved files:")
    print(TABLE_DIR / "player_shot_quality.csv")
    print(TABLE_DIR / "player_split_half_stability.csv")
    print(FIG_DIR / "player_actual_vs_expected.png")


if __name__ == "__main__":
    main()
