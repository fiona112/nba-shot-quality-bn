
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import logsumexp

from src.calibration import calculate_calibration
from src.evaluate import evaluate_model
from src.structure_learning import FEATURES, VARIABLES, prepare_data


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
    Parameters are learned with EM (MAP, add-alpha smoothing).

    Prediction marginalizes the hidden state:

        P(M=1 | x) = sum_q P(q | x) P(M=1 | q)

    where P(q | x) uses the features only, not M. A feature
    whose value was never seen in training is treated as
    unobserved and summed out.
    """

    def __init__(self, k, alpha=ALPHA, max_iter=MAX_ITER,
                 tol=TOL, seed=0):
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

    def encode(self, data):
        """Map each variable's values to integer codes (-1 = unseen)."""
        codes = {}
        for var in VARIABLES:
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
        self.states = {var: sorted(data[var].unique()) for var in VARIABLES}
        codes = self.encode(data)
        indicators = self.one_hot(codes)

        # Random soft assignments break the symmetry between states.
        rng = np.random.default_rng(self.seed)
        resp = rng.dirichlet(np.ones(self.k), size=len(data))
        self.m_step(indicators, resp)

        self.history = []
        for _ in range(self.max_iter):
            # E-step: posterior over Q for every shot.
            log_joint = self.log_joint(codes, VARIABLES)
            log_norm = logsumexp(log_joint, axis=1)
            resp = np.exp(log_joint - log_norm[:, None])

            log_likelihood = log_norm.sum()
            self.history.append({
                "log_likelihood": log_likelihood,
                "objective": self.objective(log_likelihood),
            })

            if len(self.history) > 1:
                change = (
                    self.history[-1]["objective"]
                    - self.history[-2]["objective"]
                )
                if change < self.tol * len(data):
                    self.converged = True
                    break

            # M-step: re-estimate P(Q) and every P(X | Q).
            self.m_step(indicators, resp)

        self.log_likelihood = self.history[-1]["log_likelihood"]
        self.n_iter = len(self.history)
        return self

    def n_parameters(self):
        return (self.k - 1) + self.k * sum(
            len(self.states[var]) - 1 for var in VARIABLES
        )

    def posterior(self, data):
        """P(Q | features), excluding the outcome M."""
        log_joint = self.log_joint(self.encode(data), FEATURES)
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
    fits = [
        LatentShotDifficultyBN(k, seed=seed).fit(data)
        for seed in range(N_RESTARTS)
    ]
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
        ll = [h["log_likelihood"] for h in model.history]
        ax.plot(np.arange(1, len(ll) + 1), np.asarray(ll) / 1e3,
                label=f"Restart {seed}", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Training log-likelihood (thousands)")
    ax.set_title(f"EM convergence, K = {chosen_k}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "latent_em.png", dpi=200)
    plt.close(fig)


def main():
    train_fit = prepare_data(pd.read_csv(DATA_DIR / "train_fit.csv"))
    validation = prepare_data(pd.read_csv(DATA_DIR / "validation.csv"))

    rows = []
    restarts = {}
    for k in K_VALUES:
        model, fits = fit_best(train_fit, k)
        restarts[k] = fits

        probabilities = model.predict_proba(validation)
        metrics = evaluate_model(
            f"Latent BN, K = {k}",
            validation["M"].astype(int),
            probabilities
        )
        final_ll = [m.log_likelihood for m in fits]

        rows.append({
            "k": k,
            "parameters": model.n_parameters(),
            "train_log_likelihood": model.log_likelihood,
            "bic": (-2 * model.log_likelihood
                    + model.n_parameters() * np.log(len(train_fit))),
            "restart_ll_spread": max(final_ll) - min(final_ll),
            "iterations": model.n_iter,
            "converged": model.converged,
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
    train = prepare_data(pd.read_csv(DATA_DIR / "train.csv"))
    test = prepare_data(pd.read_csv(DATA_DIR / "test.csv"))

    model, _ = fit_best(train, chosen_k)
    probabilities = model.predict_proba(test)
    y_test = test["M"].astype(int)

    metrics = evaluate_model(
        f"Latent BN, K = {chosen_k} (test)", y_test, probabilities
    )
    _, ece = calculate_calibration(y_test, probabilities)
    print(f"ECE: {ece:.4f}")

    states = model.describe_states()
    print("\n=== HIDDEN STATES (easiest to hardest) ===")
    print(states.to_string(index=False))

    hand_validation = pd.read_csv(TABLE_DIR / "structure_validation.csv")
    hand_log_loss = hand_validation.loc[
        hand_validation["model"].str.startswith("Hand"), "log_loss"
    ].iloc[0]
    plot_selection(selection, restarts[chosen_k], chosen_k, hand_log_loss)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(TABLE_DIR / "latent_em_validation.csv", index=False)
    pd.DataFrame([{
        "model": f"Latent BN (K={chosen_k})",
        **metrics,
        "ece": ece
    }]).to_csv(TABLE_DIR / "latent_em_test_metrics.csv", index=False)
    states.to_csv(TABLE_DIR / "latent_em_states.csv", index=False)

    print("\nSaved files:")
    print(TABLE_DIR / "latent_em_validation.csv")
    print(TABLE_DIR / "latent_em_test_metrics.csv")
    print(TABLE_DIR / "latent_em_states.csv")
    print(FIG_DIR / "latent_em.png")


if __name__ == "__main__":
    main()
