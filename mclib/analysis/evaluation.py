"""Policy explanation tools for the Mountain Car scenarios.

Three techniques over a common engineered feature set, all of which work on
both discrete and continuous policy grids by binning continuous actions into
three classes (left / idle / right).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeClassifier

from ..visualization.plots import plot_policy_heatmap


FEATURE_NAMES = ["pos", "vel", "pos^2", "vel^2", "pos*vel", "|vel|", "sin(3*pos)"]
POS_RANGE = (-1.2, 0.6)
VEL_RANGE = (-0.07, 0.07)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def bin_action_grid(policy_grid, deadband=0.33):
    """Map a (n_bins, n_bins) action grid to integer classes {0, 1, 2}.

    Discrete grids (already in {0, 1, 2}) pass through unchanged. Continuous
    grids (real values, typically in [-1, 1]) are binned with a symmetric
    deadband around zero: action < -deadband -> 0, |action| <= deadband -> 1,
    action > deadband -> 2.
    """
    grid = np.asarray(policy_grid)
    if np.issubdtype(grid.dtype, np.integer):
        return grid.astype(int)

    classes = np.full(grid.shape, 1, dtype=int)
    classes[grid < -deadband] = 0
    classes[grid > deadband] = 2
    return classes


def build_features(n_bins=40):
    """Build the engineered feature matrix and the matching action labels are
    supplied separately.

    Returns
    -------
    X_norm : (n_bins**2, 7) float array of z-scored features
    indices : (n_bins**2, 2) int array — (i, j) cell coordinates per row
    """
    pos_vals = np.linspace(POS_RANGE[0], POS_RANGE[1], n_bins)
    vel_vals = np.linspace(VEL_RANGE[0], VEL_RANGE[1], n_bins)

    X = np.empty((n_bins * n_bins, 7), dtype=float)
    idx = np.empty((n_bins * n_bins, 2), dtype=int)
    k = 0
    for i, p in enumerate(pos_vals):
        for j, v in enumerate(vel_vals):
            X[k] = [p, v, p * p, v * v, p * v, abs(v), np.sin(3 * p)]
            idx[k] = (i, j)
            k += 1

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / sigma
    return X_norm, idx


def _grid_targets(policy_grid, indices):
    classes = bin_action_grid(policy_grid)
    return classes[indices[:, 0], indices[:, 1]]


def _predictions_grid(preds, indices, n_bins):
    out = np.zeros((n_bins, n_bins), dtype=int)
    out[indices[:, 0], indices[:, 1]] = preds
    return out


# ---------------------------------------------------------------------------
# Technique 1 — Multinomial logistic regression (pure NumPy)
# ---------------------------------------------------------------------------


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def multinomial_logreg(X, y, n_classes=3, lr=0.05, n_iters=600, lam=0.01):
    """Multinomial logistic regression via gradient descent (pure NumPy)."""
    n, d = X.shape
    W = np.zeros((d, n_classes))
    Y_oh = np.eye(n_classes)[y]
    for _ in range(n_iters):
        probs = softmax(X @ W)
        grad = X.T @ (probs - Y_oh) / n + lam * W
        W -= lr * grad
    preds = np.argmax(X @ W, axis=1)
    acc = (preds == y).mean()
    return W, acc, preds


def fit_logreg_explanation(policy_grid, n_bins=40, scenario_name="Scenario",
                           verbose=True):
    """Fit a multinomial logistic regression to the (binned) policy grid."""
    X, indices = build_features(n_bins=n_bins)
    y = _grid_targets(policy_grid, indices)

    W, acc, preds = multinomial_logreg(X, y, n_classes=3)
    importances = np.abs(W).mean(axis=1)
    pred_grid = _predictions_grid(preds, indices, n_bins)

    if verbose:
        print(f"\n{'=' * 52}")
        print(f"[Logistic Regression] {scenario_name}")
        print(f"  Accuracy: {acc:.1%}")
        ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1],
                        reverse=True)
        print("  Feature importance (mean |coef| across classes):")
        for feat, imp in ranked:
            print(f"   {feat:12s}: {imp:.4f}")

    return {
        "technique": "logreg",
        "scenario": scenario_name,
        "accuracy": float(acc),
        "weights": W,
        "feature_names": FEATURE_NAMES,
        "importances": importances,
        "predictions_grid": pred_grid,
    }


# ---------------------------------------------------------------------------
# Technique 2 — Decision tree surrogate (sklearn)
# ---------------------------------------------------------------------------


def fit_tree_explanation(policy_grid, n_bins=40, scenario_name="Scenario",
                         max_depth=4, random_state=0, verbose=True):
    """Fit a depth-limited decision tree to the (binned) policy grid."""
    X, indices = build_features(n_bins=n_bins)
    y = _grid_targets(policy_grid, indices)

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X, y)
    preds = tree.predict(X)
    acc = (preds == y).mean()
    importances = np.asarray(tree.feature_importances_)
    pred_grid = _predictions_grid(preds.astype(int), indices, n_bins)

    if verbose:
        print(f"\n{'=' * 52}")
        print(f"[Decision Tree (depth={max_depth})] {scenario_name}")
        print(f"  Accuracy: {acc:.1%}  (leaves: {tree.get_n_leaves()})")
        ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1],
                        reverse=True)
        print("  Feature importance (sklearn impurity-based):")
        for feat, imp in ranked:
            print(f"   {feat:12s}: {imp:.4f}")

    return {
        "technique": "tree",
        "scenario": scenario_name,
        "accuracy": float(acc),
        "tree": tree,
        "feature_names": FEATURE_NAMES,
        "importances": importances,
        "predictions_grid": pred_grid,
        "max_depth": max_depth,
    }


# ---------------------------------------------------------------------------
# Technique 3 — Permutation feature importance (model-agnostic)
# ---------------------------------------------------------------------------


def permutation_importance_explanation(policy_grid, n_bins=40,
                                       scenario_name="Scenario",
                                       base="logreg", n_repeats=10, seed=0,
                                       verbose=True):
    """Permutation importance: drop in accuracy when each feature is shuffled.

    The base classifier defaults to the same multinomial logistic regression
    used by `fit_logreg_explanation` so the baseline is comparable; pass
    `base="tree"` to use the decision-tree surrogate as the scorer instead.
    """
    X, indices = build_features(n_bins=n_bins)
    y = _grid_targets(policy_grid, indices)

    if base == "logreg":
        W, baseline_acc, _ = multinomial_logreg(X, y, n_classes=3)

        def predict(Xq):
            return np.argmax(Xq @ W, axis=1)
    elif base == "tree":
        tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
        tree.fit(X, y)
        baseline_acc = (tree.predict(X) == y).mean()

        def predict(Xq):
            return tree.predict(Xq)
    else:
        raise ValueError(f"Unknown base classifier: {base!r}")

    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    drops = np.zeros((n_features, n_repeats))
    for f in range(n_features):
        for r in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, f])
            acc = (predict(X_perm) == y).mean()
            drops[f, r] = baseline_acc - acc

    importances_mean = drops.mean(axis=1)
    importances_std = drops.std(axis=1)

    if verbose:
        print(f"\n{'=' * 52}")
        print(f"[Permutation Importance, base={base}] {scenario_name}")
        print(f"  Baseline accuracy: {baseline_acc:.1%}")
        ranked = sorted(zip(FEATURE_NAMES, importances_mean, importances_std),
                        key=lambda t: t[1], reverse=True)
        print("  Mean drop in accuracy when each feature is shuffled:")
        for feat, m, s in ranked:
            print(f"   {feat:12s}: {m:+.4f} (+/- {s:.4f})")

    return {
        "technique": "permutation",
        "scenario": scenario_name,
        "base": base,
        "baseline_accuracy": float(baseline_acc),
        "feature_names": FEATURE_NAMES,
        "importances": importances_mean,
        "importances_std": importances_std,
        "n_repeats": n_repeats,
    }


# ---------------------------------------------------------------------------
# Convenience — run all three on a single policy grid
# ---------------------------------------------------------------------------


def run_all_explanations(policy_grid, n_bins=40, scenario_name="Scenario",
                         tree_max_depth=4, n_repeats=10, seed=0, verbose=True):
    """Run the three explanation techniques and return their results."""
    return {
        "scenario": scenario_name,
        "logreg": fit_logreg_explanation(
            policy_grid, n_bins=n_bins, scenario_name=scenario_name,
            verbose=verbose,
        ),
        "tree": fit_tree_explanation(
            policy_grid, n_bins=n_bins, scenario_name=scenario_name,
            max_depth=tree_max_depth, verbose=verbose,
        ),
        "permutation": permutation_importance_explanation(
            policy_grid, n_bins=n_bins, scenario_name=scenario_name,
            base="logreg", n_repeats=n_repeats, seed=seed, verbose=verbose,
        ),
    }


# ---------------------------------------------------------------------------
# Backwards-compat alias for the old API
# ---------------------------------------------------------------------------


def explain_policy_regression(policy_grid, n_bins=40, scenario_name="Scenario"):
    """Alias for the original API: prints summary, returns (acc, W, names)."""
    res = fit_logreg_explanation(policy_grid, n_bins=n_bins,
                                 scenario_name=scenario_name, verbose=True)
    return res["accuracy"], res["weights"], res["feature_names"]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_feature_importance_bars(results_by_scenario, technique="logreg",
                                 ax=None, title=None):
    """Grouped bar chart of feature importances across scenarios.

    Parameters
    ----------
    results_by_scenario : dict[str, dict]
        Mapping scenario name -> result dict produced by
        `run_all_explanations` (or one of the per-technique fit functions).
        If values are the full `run_all_explanations` dict, the requested
        `technique` key is read out automatically.
    technique : {"logreg", "tree", "permutation"}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))

    scenarios = list(results_by_scenario.keys())
    importances = []
    for sc in scenarios:
        entry = results_by_scenario[sc]
        if technique in entry:
            entry = entry[technique]
        importances.append(np.asarray(entry["importances"]))

    n_features = len(FEATURE_NAMES)
    n_scenarios = len(scenarios)
    width = 0.8 / max(n_scenarios, 1)
    x = np.arange(n_features)

    for k, (sc, imp) in enumerate(zip(scenarios, importances)):
        offset = (k - (n_scenarios - 1) / 2) * width
        ax.bar(x + offset, imp, width=width, label=sc)

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right")
    ax.set_ylabel({
        "logreg": "mean |coef|",
        "tree": "impurity importance",
        "permutation": "accuracy drop",
    }.get(technique, "importance"))
    ax.set_title(title or f"Feature importance ({technique})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return ax


def plot_decision_boundary_overlay(policy_grid, predictions_grid, n_bins=40,
                                   ax=None, title="Surrogate vs. policy"):
    """Plot the (binned) policy heatmap with surrogate disagreements overlaid.

    Cells where the surrogate predicts a different action than the actual
    policy are marked. Use the `predictions_grid` field of any result dict
    from this module.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    binned = bin_action_grid(policy_grid)
    plot_policy_heatmap(binned, n_bins=n_bins, title=title, ax=ax)

    pos_edges = np.linspace(POS_RANGE[0], POS_RANGE[1], n_bins + 1)
    vel_edges = np.linspace(VEL_RANGE[0], VEL_RANGE[1], n_bins + 1)
    pos_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])
    vel_centers = 0.5 * (vel_edges[:-1] + vel_edges[1:])
    PP, VV = np.meshgrid(pos_centers, vel_centers, indexing="ij")

    disagree = binned != predictions_grid
    ax.scatter(PP[disagree], VV[disagree], s=6, c="black", marker="x",
               linewidths=0.6)

    legend_patches = [
        mpatches.Patch(color="red", label="Thrust Left"),
        mpatches.Patch(color="cyan", label="Idle"),
        mpatches.Patch(color="green", label="Thrust Right"),
        mpatches.Patch(color="black", label="Surrogate disagrees"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
    return ax
