import numpy as np


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def multinomial_logreg(X, y, n_classes=3, lr=0.05, n_iters=600, lam=0.01):
    """
    Multinomial logistic regression via gradient descent (pure NumPy).
    Returns weight matrix W (features x classes) and accuracy.
    """
    n, d = X.shape
    W = np.zeros((d, n_classes))
    Y_oh = np.eye(n_classes)[y]
    for _ in range(n_iters):
        probs = softmax(X @ W)
        grad= X.T @ (probs - Y_oh) / n + lam * W
        W -= lr * grad
    preds = np.argmax(X @ W, axis=1)
    acc = (preds == y).mean()
    return W, acc


def explain_policy_regression(policy_grid, n_bins=40, scenario_name="Scenario"):
    """
    Fit a multinomial logistic regression to the policy grid and report
    feature importances.

    Features: [pos, vel, pos^2, vel^2, pos*vel, |vel|, sin(3*pos)]
    Target: greedy action (0=left, 1=idle, 2=right)
    """
    pos_vals = np.linspace(-1.2, 0.6, n_bins)
    vel_vals = np.linspace(-0.07, 0.07, n_bins)

    X, y = [], []
    for i, p in enumerate(pos_vals):
        for j, v in enumerate(vel_vals):
            X.append([p, v, p**2, v**2, p * v, abs(v), np.sin(3 * p)])
            y.append(policy_grid[i, j])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    mu  = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / sigma

    W, acc = multinomial_logreg(X_norm, y)

    feature_names = ["pos", "vel", "pos^2", "vel^2", "pos*vel", "|vel|", "sin(3*pos)"]

    print(f"\n{'='*52}")
    print(f"Policy Explanation: {scenario_name}")
    print(f"Logistic Regression Accuracy: {acc:.1%}")
    mean_abs = np.abs(W).mean(axis=1)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    print("Feature importance (mean |coef| across classes):")
    for feat, imp in ranked:
        print(f" {feat:12s}: {imp:.4f}")

    return acc, W, feature_names
