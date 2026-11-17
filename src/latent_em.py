"""Anna He's latent-class BN, integrated with verified experiment provenance.

Original method and K/restart protocol: team PR #4 (commit c816b57f).
Identical observed configurations are compressed into weighted sufficient counts;
the original per-shot seeded initialization and MAP-EM updates are preserved.
"""

from pathlib import Path
from time import perf_counter
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import logsumexp

from src.calibration import calculate_calibration
from src.ci_sensitivity import log_losses, paired_game_bootstrap
from src.evaluate import evaluate_model
from src.final_evaluation import cohort_summary, load_final_data
from src.structure_learning import (
    FEATURES, HAND_PARENTS, VARIABLES, ConditionalMakeModel, prepare_data,
)
from src.validation import load_validation_data, write_validation_metadata


DATA_DIR = Path("data/processed")
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")

K_VALUES = list(range(2, 11)) + [12, 14, 16, 18, 20]
N_RESTARTS = 5
ALPHA = 1.0
MAX_ITER = 1000
TOL = 1e-7


class LatentShotDifficultyBN:
    """
    Latent-class Bayesian network:

        Q -> T, D, C, S, B, H, M

    Q is a hidden "shot difficulty" state with K values.
    Every observed variable, including the outcome M, is
    conditionally independent of the others given Q.
    Parameters are learned with EM (MAP, add-alpha smoothing). The associated
    symmetric Dirichlet concentration is alpha + 1, since its log-prior exponent
    is alpha; this differs from a Dirichlet(alpha) posterior-mean estimate.

    Prediction marginalizes the hidden state:

        P(M=1 | x) = sum_q P(q | x) P(M=1 | q)

    where P(q | x) uses the features only, not M. A feature
    whose value was never seen in training is treated as
    unobserved and summed out.
    """

    def __init__(self, k, alpha=ALPHA, max_iter=MAX_ITER,
                 tol=TOL, seed=0):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer.")
        if (alpha <= 0 or not np.isfinite(alpha) or not isinstance(max_iter, (int, np.integer))
                or max_iter < 1 or tol < 0 or not np.isfinite(tol)):
            raise ValueError("Require finite alpha > 0, max_iter >= 1, and tol >= 0.")
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.states = {}
        self.log_prior = None
        self.log_cpts = {}
        self.history = []
        self.converged = False

    def encode(self, data, variables=VARIABLES):
        """Map each variable's values to integer codes (-1 = unseen)."""
        codes = {}
        for var in variables:
            lookup = {s: i for i, s in enumerate(self.states[var])}
            codes[var] = (
                data[var].map(lookup).fillna(-1).astype(int).to_numpy()
            )
        return codes

    def log_joint(self, codes, variables):
        """log P(q) + sum_v log P(x_v | q) for every row and state."""
        n = len(next(iter(codes.values())))
        log_joint = np.tile(self.log_prior, (n, 1))

        for var in variables:
            x = codes[var]
            seen = x >= 0
            log_joint[seen] += self.log_cpts[var][:, x[seen]].T

        return log_joint

    def one_hot(self, codes):
        """One indicator matrix (rows x states) per variable."""
        return {
            var: np.eye(len(self.states[var]))[codes[var]]
            for var in VARIABLES
        }

    def m_step(self, indicators, resp):
        weights = resp.sum(axis=0)
        self.log_prior = np.log(
            (weights + self.alpha)
            / (weights.sum() + self.k * self.alpha)
        )

        for var in VARIABLES:
            n_states = len(self.states[var])
            # counts[q, s] = expected number of rows with Q=q and var=s
            counts = resp.T @ indicators[var]
            self.log_cpts[var] = np.log(
                (counts + self.alpha)
                / (weights[:, None] + n_states * self.alpha)
            )

    def objective(self, log_likelihood):
        """
        Log posterior that MAP-EM increases at every iteration:
        data log-likelihood plus the Dirichlet log prior.
        """
        log_prior = self.alpha * self.log_prior.sum()
        for cpt in self.log_cpts.values():
            log_prior += self.alpha * cpt.sum()
        return log_likelihood + log_prior

    def fit(self, data):
        if data.empty or not set(VARIABLES).issubset(data) or data[VARIABLES].isna().any().any():
            raise ValueError("EM training requires nonempty, complete observed model variables.")
        if not data["M"].isin(["0", "1"]).all():
            raise ValueError("Prepare training targets as binary categorical strings.")
        self.states = {var: sorted(data[var].unique()) for var in VARIABLES}
        self.states["M"] = ["0", "1"]
        self.converged = False
        self.log_cpts = {}
        codes = self.encode(data)
        indicators = self.one_hot(codes)

        # Random soft assignments break the symmetry between states.
        rng = np.random.default_rng(self.seed)
        resp = rng.dirichlet(np.ones(self.k), size=len(data))
        self.m_step(indicators, resp)

        # All identical observed tuples have identical E-step responsibilities.
        # Keep frequencies so this is the same shot-weighted likelihood and
        # M-step as uncompressed EM, including the original initialization above.
        patterns, frequencies = np.unique(
            np.column_stack([codes[var] for var in VARIABLES]), axis=0, return_counts=True,
        )
        codes = {var: patterns[:, i] for i, var in enumerate(VARIABLES)}
        indicators = self.one_hot(codes)
        self.training_shots = len(data)
        self.observed_configurations = len(patterns)
        self.history = []
        for iteration in range(self.max_iter):
            # E-step: posterior over Q for every shot.
            log_joint = self.log_joint(codes, VARIABLES)
            log_norm = logsumexp(log_joint, axis=1)
            resp = np.exp(log_joint - log_norm[:, None])

            log_likelihood = float(log_norm @ frequencies)
            self.history.append({
                "log_likelihood": log_likelihood,
                "objective": self.objective(log_likelihood),
            })

            if len(self.history) > 1:
                change = (
                    self.history[-1]["objective"]
                    - self.history[-2]["objective"]
                )
                numerical_tolerance = 1e-10 * max(1.0, abs(self.history[-2]["objective"]))
                if change < -numerical_tolerance:
                    raise RuntimeError("MAP-EM objective decreased beyond numerical tolerance.")
                if change < self.tol * len(data):
                    self.converged = True
                    break

            # M-step: re-estimate P(Q) and every P(X | Q).
            # Do not return an unrecorded M-step at the iteration cap.
            if iteration + 1 < self.max_iter:
                self.m_step(indicators, resp * frequencies[:, None])

        self.log_likelihood = self.history[-1]["log_likelihood"]
        self.n_iter = len(self.history)
        return self

    def n_parameters(self):
        return (self.k - 1) + self.k * sum(
            len(self.states[var]) - 1 for var in VARIABLES
        )

    def posterior(self, data):
        """P(Q | features), excluding the outcome M."""
        if self.log_prior is None:
            raise RuntimeError("Fit the model before predicting.")
        log_joint = self.log_joint(self.encode(data, FEATURES), FEATURES)
        return np.exp(
            log_joint - logsumexp(log_joint, axis=1)[:, None]
        )

    def predict_proba(self, data):
        made = self.states["M"].index("1")
        p_make_given_q = np.exp(self.log_cpts["M"][:, made])
        return self.posterior(data) @ p_make_given_q

    def describe_states(self):
        """One row per hidden state, sorted from easiest to hardest."""
        made = self.states["M"].index("1")
        rows = []

        for q in range(self.k):
            row = {
                "weight": np.exp(self.log_prior[q]),
                "p_make": np.exp(self.log_cpts["M"][q, made]),
            }
            for var in FEATURES:
                probs = np.exp(self.log_cpts[var][q])
                top = probs.argmax()
                row[f"{var}_mode"] = self.states[var][top]
                row[f"{var}_mode_prob"] = probs[top]
            rows.append(row)

        table = pd.DataFrame(rows).sort_values("p_make", ascending=False)
        table.insert(0, "state", range(1, self.k + 1))
        return table.reset_index(drop=True)


def fit_best(data, k):
    """Run several random restarts and keep the highest objective."""
    fits = []
    for seed in range(N_RESTARTS):
        start = perf_counter()
        model = LatentShotDifficultyBN(
            k, seed=seed, alpha=ALPHA, max_iter=MAX_ITER, tol=TOL,
        ).fit(data)
        model.fit_seconds = perf_counter() - start
        fits.append(model)
    best = max(fits, key=lambda m: m.history[-1]["objective"])
    return best, fits


def plot_selection(selection, best_fits, chosen_k, hand_log_loss):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(selection["k"], selection["val_log_loss"], marker="o",
            label="Latent-class BN")
    ax.axhline(hand_log_loss, linestyle="--", color="gray",
               label="Hand-designed BN")
    ax.set_xlabel("Number of hidden states K")
    ax.set_ylabel("Validation log loss")
    ax.set_title("Model selection")
    ax.legend()

    ax = axes[1]
    for seed, model in enumerate(best_fits):
        objective = [h["objective"] for h in model.history]
        ax.plot(np.arange(1, len(objective) + 1), np.asarray(objective) / 1e3,
                label=f"Restart {seed}", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Training MAP objective (thousands)")
    ax.set_title(f"EM convergence, K = {chosen_k}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "latent_em.png", dpi=200)
    plt.close(fig)


def load_experiment_data():
    """Check all four splits and verify validation's complete outer-train union."""
    train_fit, validation, inner_cohort = load_validation_data(DATA_DIR)
    train, test, cohort = load_final_data(DATA_DIR)
    if inner_cohort["preprocessing_report_sha256"] != cohort["preprocessing_report_sha256"]:
        raise ValueError("Preprocessing report changed while loading the EM cohorts.")
    combined = pd.concat([train_fit, validation], ignore_index=True)
    columns = ["RAW_ROW_ID", "GAME_ID", *VARIABLES]
    try:
        pd.testing.assert_frame_equal(
            combined[columns].sort_values("RAW_ROW_ID").reset_index(drop=True),
            train[columns].sort_values("RAW_ROW_ID").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("Training-fit plus validation must exactly reconstruct outer training.") from error
    cohort["inputs"].update({
        name: {**description, "file": str(DATA_DIR / description["file"])}
        for name, description in inner_cohort["inputs"].items()
    })
    cohort["cohorts"] = {
        name: cohort_summary(frame) for name, frame in (
            ("train_fit", train_fit), ("validation", validation), ("train", train), ("test", test),
        )
    }
    return train_fit, validation, train, test, cohort


def restart_diagnostics(stage, fits, chosen_model):
    """Compact diagnostics plus detailed history saved outside version control."""
    summaries, traces = [], []
    for model in fits:
        changes = np.diff([item["objective"] for item in model.history])
        summaries.append({
            "stage": stage, "k": model.k, "seed": model.seed,
            "selected_restart": model is chosen_model,
            "iterations": model.n_iter, "converged": model.converged,
            "hit_iteration_cap": not model.converged and model.n_iter == model.max_iter,
            "log_likelihood": model.log_likelihood,
            "map_objective": model.history[-1]["objective"],
            "minimum_objective_change": float(changes.min()) if len(changes) else None,
            "fit_seconds": model.fit_seconds,
            "training_shots": model.training_shots,
            "observed_configurations": model.observed_configurations,
        })
        traces.extend({
            "stage": stage, "k": model.k, "seed": model.seed,
            "iteration": index + 1, **item,
        } for index, item in enumerate(model.history))
    return summaries, traces


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_fit_raw, validation_raw, train_raw, test_raw, cohort = load_experiment_data()
    train_fit, validation = prepare_data(train_fit_raw), prepare_data(validation_raw)

    rows = []
    restarts = {}
    diagnostics, traces = [], []
    for k in K_VALUES:
        model, fits = fit_best(train_fit, k)
        restarts[k] = fits
        summary, history = restart_diagnostics("validation_fit", fits, model)
        diagnostics.extend(summary)
        traces.extend(history)

        probabilities = model.predict_proba(validation)
        metrics = evaluate_model(
            f"Latent BN, K = {k}",
            validation["M"].astype(int),
            probabilities
        )
        final_ll = [m.log_likelihood for m in fits]
        print(f"K={k}: best seed={model.seed}, iterations={model.n_iter}, "
              f"converged={model.converged}, total fit seconds={sum(m.fit_seconds for m in fits):.2f}", flush=True)

        rows.append({
            "k": k,
            "training_shots": len(train_fit),
            "validation_shots": len(validation),
            "selected_seed": model.seed,
            "parameters": model.n_parameters(),
            "train_log_likelihood": model.log_likelihood,
            "train_map_objective": model.history[-1]["objective"],
            "bic": (-2 * model.log_likelihood
                    + model.n_parameters() * np.log(len(train_fit))),
            "restart_ll_spread": max(final_ll) - min(final_ll),
            "iterations": model.n_iter,
            "converged": model.converged,
            "converged_restarts": sum(fit.converged for fit in fits),
            "restart_fit_seconds": sum(fit.fit_seconds for fit in fits),
            "val_log_loss": metrics["log_loss"],
            "val_brier_score": metrics["brier_score"],
            "val_accuracy": metrics["accuracy"],
        })

    selection = pd.DataFrame(rows)
    chosen_k = int(
        selection.loc[selection["val_log_loss"].idxmin(), "k"]
    )

    print("\n=== MODEL SELECTION ===")
    print(selection.to_string(index=False))
    print("\nChosen K (lowest validation log loss):", chosen_k)

    # Refit the chosen K on all training games; score the test games once.
    train, test = prepare_data(train_raw), prepare_data(test_raw)

    model, refit_restarts = fit_best(train, chosen_k)
    summary, history = restart_diagnostics("outer_train_refit", refit_restarts, model)
    diagnostics.extend(summary)
    traces.extend(history)
    probabilities = model.predict_proba(test)
    y_test = test["M"].astype(int)

    metrics = evaluate_model(
        f"Latent BN, K = {chosen_k} (test)", y_test, probabilities
    )
    calibration_bins, ece = calculate_calibration(y_test, probabilities)
    print(f"ECE: {ece:.4f}")

    states = model.describe_states()
    print("\n=== HIDDEN STATES (easiest to hardest) ===")
    print(states.to_string(index=False))

    # Recompute the fixed hand reference on these checked cohorts rather than
    # depending on a possibly stale saved validation table.
    hand = ConditionalMakeModel(HAND_PARENTS, alpha=1.0).fit(train_fit)
    hand_log_loss = float(log_losses(validation["M"].astype(int), hand.predict_proba(validation)).mean())
    plot_selection(selection, restarts[chosen_k], chosen_k, hand_log_loss)

    outputs = []
    selection_path = TABLE_DIR / "latent_em_validation.csv"
    selection.to_csv(selection_path, index=False)
    outputs.append(selection_path)
    chosen_validation = selection.loc[selection["k"].eq(chosen_k)].iloc[0]
    pd.DataFrame([{
        "model": f"Latent BN (K={chosen_k})",
        "k": chosen_k, "selected_seed": model.seed,
        "validation_selected_seed": int(chosen_validation["selected_seed"]),
        "validation_log_loss": float(chosen_validation["val_log_loss"]),
        "training_shots": len(train), "evaluation_shots": len(test),
        "training_games": int(train_raw["GAME_ID"].nunique()),
        "evaluation_games": int(test_raw["GAME_ID"].nunique()),
        "train_log_likelihood": model.log_likelihood,
        "train_map_objective": model.history[-1]["objective"],
        "iterations": model.n_iter, "converged": model.converged,
        "hit_iteration_cap": not model.converged and model.n_iter == model.max_iter,
        **metrics,
        "ece": ece
    }]).to_csv(TABLE_DIR / "latent_em_test_metrics.csv", index=False)
    states.to_csv(TABLE_DIR / "latent_em_states.csv", index=False)
    calibration_bins.to_csv(TABLE_DIR / "latent_em_calibration_bins.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(TABLE_DIR / "latent_em_restarts.csv", index=False)
    trace_path = DATA_DIR / "latent_em_trace.csv"
    pd.DataFrame(traces).to_csv(trace_path, index=False)
    prediction_path = DATA_DIR / "latent_em_test_predictions.csv"
    predictions = test_raw[["RAW_ROW_ID", "GAME_ID"]].reset_index(drop=True).copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["Latent BN"] = probabilities
    hand = ConditionalMakeModel(HAND_PARENTS, alpha=1.0).fit(train)
    predictions["Hand-Designed BN"] = hand.predict_proba(test)
    predictions.to_csv(prediction_path, index=False)
    paired = {
        "comparison": "Latent BN versus fixed hand BN", "k": chosen_k,
        "positive_favors": "Latent BN",
        **paired_game_bootstrap(
            log_losses(y_test, predictions["Hand-Designed BN"]) - log_losses(y_test, probabilities),
            predictions["GAME_ID"],
        ),
    }
    pd.DataFrame([paired]).to_csv(TABLE_DIR / "latent_em_paired_comparison.csv", index=False)
    outputs.extend(TABLE_DIR / f"latent_em_{name}.csv" for name in (
        "test_metrics", "states", "calibration_bins", "restarts", "paired_comparison",
    ))
    outputs.extend([trace_path, prediction_path, FIG_DIR / "latent_em.png"])
    write_validation_metadata(
        TABLE_DIR / "latent_em_metadata.json", "latent_em", cohort,
        {
            "method_attribution": "Anna He; original team PR #4, commit c816b57f76e4f8a91460e8d1ae9ced1d6ba13530",
            "graph": "Q -> T,D,C,S,B,H,M; observed nodes independent given Q",
            "k_values": K_VALUES, "seeds": list(range(N_RESTARTS)),
            "alpha_pseudocount": ALPHA, "dirichlet_concentration": ALPHA + 1,
            "max_iter": MAX_ITER, "tol_per_shot": TOL,
            "stopping": "MAP objective improvement < tol * training_shots; material decrease raises an error; cap is recorded",
            "restart_selection": "highest training MAP objective for each K; no validation-based restart choice",
            "k_selection": "lowest validation log loss; ties use first K in fixed ascending grid",
            "selected_k": chosen_k,
            "refit": "same fixed seeds and settings on complete outer training, then evaluate unchanged original test rows",
            "prediction": "marginalize Q conditional on features only; unseen features summed out; target never used",
            "compression": "frequency-weighted distinct observed tuples; seeded initialization remains per-shot",
            "numerical_environment": {name: os.environ.get(name) for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            )},
            "bic": "descriptive formula evaluated at MAP fit, not used for selection; finite mixtures need not satisfy regular BIC assumptions",
            "calibration_bins": 10, "calibration_binning": "same equal-width bins as common final evaluation",
            "interval": "1000 paired game-cluster percentile replicates, seed 20261117; positive hand loss minus latent loss favors latent",
            "limitations": "Original test was inspected in earlier work. Selected K is exploratory validation selection; intervals condition on fixed fits and omit selection/refit uncertainty and player dependence across games. Q labels are statistical shot configurations, not identified causal difficulty or shooter skill.",
        }, outputs,
    )

    print("\nSaved files:")
    print(TABLE_DIR / "latent_em_validation.csv")
    print(TABLE_DIR / "latent_em_test_metrics.csv")
    print(TABLE_DIR / "latent_em_states.csv")
    print(FIG_DIR / "latent_em.png")


if __name__ == "__main__":
    main()
