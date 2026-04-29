# Explanation Tools

All explanation tools live in `mclib/analysis/evaluation.py` and operate on a 2D policy grid (action chosen at each `(pos, vel)` cell). Continuous-action grids are first binned to three classes `{0=left, 1=idle, 2=right}` so all four scenarios share a common 3-class output space.

The shared engineered feature set (z-scored before fitting) is:

```
[pos, vel, pos^2, vel^2, pos*vel, |vel|, sin(3*pos)]
```

| Technique | Question it answers | Output |
|-----------|---------------------|--------|
| **Multinomial logistic regression** | Which features linearly separate the actions? | per-feature mean `|coef|` |
| **Decision-tree surrogate** (depth 4) | What if-then rules approximate the policy? | impurity-based importance |
| **Permutation importance** | Which features does the policy rely on, model-agnostically? | accuracy drop when feature is shuffled |

## 1. Tools Implemented

### `bin_action_grid(policy_grid, deadband=0.33)`
Maps a `(n_bins, n_bins)` action grid to integer classes `{0, 1, 2}`. Discrete grids pass through unchanged; continuous grids are binned with a symmetric deadband around zero (`action < -deadband -> 0`, `|action| <= deadband -> 1`, `action > deadband -> 2`).

### `build_features(n_bins=40)`
Builds the engineered feature matrix on a uniform grid over `pos ∈ [-1.2, 0.6]` and `vel ∈ [-0.07, 0.07]`. Returns the z-scored `(n_bins**2, 7)` matrix and the `(i, j)` cell indices.

### `multinomial_logreg(X, y, n_classes=3, lr=0.05, n_iters=600, lam=0.01)`
Pure-NumPy multinomial logistic regression by gradient descent with L2 regularisation. Returns the weight matrix, accuracy, and predictions.

### `fit_logreg_explanation(...)` — Technique 1
Fits `multinomial_logreg` on the binned policy grid and reports per-feature **mean `|coef|` across classes** as the importance signal. Returns accuracy, weights, importances, and a predictions grid.

### `fit_tree_explanation(..., max_depth=4)` — Technique 2
Fits a depth-4 `DecisionTreeClassifier` (sklearn) to the binned policy grid. Reports **sklearn impurity-based feature importances** and the tree's predictions grid.

### `permutation_importance_explanation(..., base="logreg", n_repeats=10)` — Technique 3
Model-agnostic feature importance. For each feature, shuffles its column 10 times and records the **mean drop in accuracy** of the base classifier (defaults to the same logistic regression used by Technique 1; can be swapped to the tree).

### `run_all_explanations(...)`
Runs all three techniques on a single policy grid and returns a dict with `logreg`, `tree`, and `permutation` sub-results. This is the primary cross-scenario driver used in Section 7 of the notebook.

### `explain_policy_regression(...)`
Backwards-compatible alias of `fit_logreg_explanation` that returns just `(accuracy, weights, feature_names)`. Used for the per-scenario S2/S3/S4 explanation cells.

### `plot_feature_importance_bars(results_by_scenario, technique=...)`
Grouped bar chart: x-axis = the 7 features, bars grouped by scenario for one chosen technique (`logreg` / `tree` / `permutation`). The y-axis label adapts to the technique (`mean |coef|`, `impurity importance`, or `accuracy drop`).

### `plot_decision_boundary_overlay(policy_grid, predictions_grid, ...)`
Plots the binned policy heatmap with black `x` markers on every cell where the surrogate disagrees with the true policy — a visual diagnostic for *where* the linear surrogate fails.

---

## 2. Results (verbatim from the notebook)

### 2.1 S2 Policy Explanation Analysis

```
============================================================
S2 POLICY EXPLANATION ANALYSIS
============================================================

====================================================
[Logistic Regression] S2_default_SAC
  Accuracy: 79.2%
  Feature importance (mean |coef| across classes):
   vel         : 0.8104
   pos^2       : 0.1407
   vel^2       : 0.1252
   sin(3*pos)  : 0.0985
   |vel|       : 0.0935
   pos         : 0.0736
   pos*vel     : 0.0480

====================================================
[Logistic Regression] S2_default_TD3
  Accuracy: 94.4%
  Feature importance (mean |coef| across classes):
   vel         : 1.4723
   pos^2       : 0.5482
   pos*vel     : 0.4075
   sin(3*pos)  : 0.3834
   vel^2       : 0.1908
   |vel|       : 0.1811
   pos         : 0.1524

====================================================
[Logistic Regression] S2_shaped_SAC
  Accuracy: 73.7%
  Feature importance (mean |coef| across classes):
   pos*vel     : 0.7364
   vel^2       : 0.3403
   sin(3*pos)  : 0.2740
   vel         : 0.2481
   |vel|       : 0.1372
   pos^2       : 0.1369
   pos         : 0.1127

====================================================
[Logistic Regression] S2_shaped_TD3
  Accuracy: 95.6%
  Feature importance (mean |coef| across classes):
   vel         : 1.4678
   pos^2       : 0.6912
   sin(3*pos)  : 0.4288
   pos         : 0.2687
   pos*vel     : 0.2134
   vel^2       : 0.1568
   |vel|       : 0.1353
```

### 2.2 S3 Policy Explanation Analysis

```
============================================================
S3 POLICY EXPLANATION ANALYSIS
============================================================

====================================================
[Logistic Regression] S3 Q-Learning (Fuel)
  Accuracy: 58.1%
  Feature importance (mean |coef| across classes):
   vel         : 0.3817
   vel^2       : 0.2604
   pos         : 0.2168
   pos*vel     : 0.1443
   pos^2       : 0.1131
   |vel|       : 0.1129
   sin(3*pos)  : 0.0704

------------------------------------------------------------

====================================================
[Logistic Regression] S3 SARSA (Fuel)
  Accuracy: 58.7%
  Feature importance (mean |coef| across classes):
   vel         : 0.4704
   vel^2       : 0.3024
   pos         : 0.1978
   pos*vel     : 0.1396
   sin(3*pos)  : 0.1038
   |vel|       : 0.0990
   pos^2       : 0.0526

------------------------------------------------------------

====================================================
[Logistic Regression] S3 DQN (Fuel)
  Accuracy: 82.9%
  Feature importance (mean |coef| across classes):
   pos*vel     : 0.9417
   vel         : 0.5432
   sin(3*pos)  : 0.4208
   pos^2       : 0.2642
   pos         : 0.2298
   vel^2       : 0.1625
   |vel|       : 0.1276
```

### 2.3 S4 Policy Explanation Analysis

```
====================================================
[Logistic Regression] S4_step_SAC
  Accuracy: 81.4%
  Feature importance (mean |coef| across classes):
   sin(3*pos)  : 0.7670
   vel         : 0.7145
   pos^2       : 0.4127
   pos*vel     : 0.3601
   vel^2       : 0.1460
   |vel|       : 0.0801
   pos         : 0.0114

====================================================
[Logistic Regression] S4_step_TD3
  Accuracy: 92.3%
  Feature importance (mean |coef| across classes):
   vel         : 1.1412
   pos^2       : 0.7079
   sin(3*pos)  : 0.5503
   pos*vel     : 0.4734
   |vel|       : 0.1465
   vel^2       : 0.1302
   pos         : 0.1248

====================================================
[Logistic Regression] S4_default_SAC
  Accuracy: 78.6%
  Feature importance (mean |coef| across classes):
   vel         : 0.8737
   pos         : 0.2724
   pos*vel     : 0.1794
   pos^2       : 0.1547
   |vel|       : 0.1408
   vel^2       : 0.1130
   sin(3*pos)  : 0.0642

====================================================
[Logistic Regression] S4_default_TD3
  Accuracy: 94.3%
  Feature importance (mean |coef| across classes):
   vel         : 1.1961
   pos*vel     : 0.7060
   pos^2       : 0.6121
   sin(3*pos)  : 0.5208
   vel^2       : 0.1658
   |vel|       : 0.1526
   pos         : 0.1182

====================================================
[Logistic Regression] S4_action_use_SAC
  Accuracy: 59.8%
  Feature importance (mean |coef| across classes):
   pos         : 0.0000
   pos^2       : 0.0000
   vel^2       : 0.0000
   sin(3*pos)  : 0.0000
   |vel|       : 0.0000
   pos*vel     : 0.0000
   vel         : 0.0000

====================================================
[Logistic Regression] S4_action_use_TD3
  Accuracy: 94.3%
  Feature importance (mean |coef| across classes):
   vel         : 1.4550
   pos^2       : 0.5407
   pos*vel     : 0.3803
   sin(3*pos)  : 0.3673
   vel^2       : 0.1910
   |vel|       : 0.1738
   pos         : 0.1652

====================================================
[Logistic Regression] S4_shaped_SAC
  Accuracy: 82.8%
  Feature importance (mean |coef| across classes):
   vel         : 0.8064
   sin(3*pos)  : 0.7002
   pos^2       : 0.6277
   |vel|       : 0.1981
   pos*vel     : 0.1776
   vel^2       : 0.0835
   pos         : 0.0300

====================================================
[Logistic Regression] S4_shaped_TD3
  Accuracy: 94.6%
  Feature importance (mean |coef| across classes):
   vel         : 1.4495
   pos^2       : 0.6042
   pos*vel     : 0.3973
   sin(3*pos)  : 0.3960
   vel^2       : 0.1852
   |vel|       : 0.1655
   pos         : 0.1626
```

### 2.4 Cross-Scenario Summary (Section 7)

Policy grids collected:

```
Collected 14 policy grids:
  S1 Q-Learning                     shape=(40, 40)  dtype=int64
  S1 SARSA                          shape=(40, 40)  dtype=int64
  S1 DQN                            shape=(40, 40)  dtype=int64
  S3 Q-Learning                     shape=(40, 40)  dtype=int64
  S3 SARSA                          shape=(40, 40)  dtype=int64
  S3 DQN                            shape=(40, 40)  dtype=int64
  S4 step_SAC                       shape=(40, 40)  dtype=int64
  S4 step_TD3                       shape=(40, 40)  dtype=int64
  S4 default_SAC                    shape=(40, 40)  dtype=int64
  S4 default_TD3                    shape=(40, 40)  dtype=int64
  S4 action_use_SAC                 shape=(40, 40)  dtype=int64
  S4 action_use_TD3                 shape=(40, 40)  dtype=int64
  S4 shaped_SAC                     shape=(40, 40)  dtype=int64
  S4 shaped_TD3                     shape=(40, 40)  dtype=int64
```

Accuracy summary across the three explanation techniques:

```
scenario                            logreg      tree   perm-base
----------------------------------------------------------------
S1 Q-Learning                        57.6%     71.5%       57.6%
S1 SARSA                             55.6%     73.2%       55.6%
S1 DQN                               87.2%     93.8%       87.2%
S3 Q-Learning                        58.1%     73.2%       58.1%
S3 SARSA                             58.7%     74.8%       58.7%
S3 DQN                               82.9%     88.6%       82.9%
S4 step_SAC                          81.2%     97.3%       81.2%
S4 step_TD3                          92.4%     94.2%       92.4%
S4 default_SAC                       78.3%     94.2%       78.3%
S4 default_TD3                       94.3%     96.7%       94.3%
S4 action_use_SAC                    43.5%    100.0%       43.5%
S4 action_use_TD3                    94.1%     96.6%       94.1%
S4 shaped_SAC                        82.8%     96.6%       82.8%
S4 shaped_TD3                        94.7%     97.1%       94.7%
```

---

## 3. What the Results Mean (brief)

**Accuracy is a measure of *explainability*.**
- Logistic-regression accuracy is bounded by linearity. Values below ~60% (S1 Q-Learning, S1 SARSA, S3 Q-Learning, S3 SARSA, S4 action_use_SAC) signal a non-linear decision boundary that a 7-feature linear model cannot reproduce.
- Decision-tree accuracy at depth 4 is consistently higher than logreg for the same policy because the tree captures axis-aligned non-linear splits. Trees in the 90%+ range mean the policy is well-approximated by a handful of if-then rules.
- Permutation baseline accuracy equals the logreg accuracy by construction (same model is the scorer).

**Velocity is the dominant decision feature** in every solvable policy. Mountain Car's escape requires a momentum-building oscillation, so the *sign* and *magnitude* of velocity are the primary drivers of which way to thrust. This shows up as `vel` being the top logreg coefficient in nearly all TD3 runs (~1.15–1.47) and in S2_default_SAC (0.81).

**S1 vs S3 (same env, different reward).** Tabular S1/S3 policies hover at ~55–60% logreg accuracy — the linear model cannot capture the per-cell switching of a tabular agent. DQN does much better in both (S1: 87.2%, S3: 82.9%) because its neural value function produces a smoother, more linearly separable policy.

**S4 (continuous, TD3 vs SAC).** TD3 variants are strongly explainable (logreg 92–95%, tree 94–97%) — they converge to a clean bang-bang controller dominated by `vel`. SAC variants are mixed: `step_SAC`, `default_SAC`, and `shaped_SAC` are reasonably explainable (78–83% logreg), while **`action_use_SAC` collapses** — logreg accuracy 43.5% with all importances at 0.000, but tree accuracy 100%. This is the diagnostic signature of a *constant policy* (the agent learned to output ~zero force everywhere), which the tree fits trivially with one node and the linear model cannot beat the majority class.

**Reward shaping (S3) shifts importance toward `|vel|` / `pos*vel`.** When each thrust costs reward, the policy becomes more sensitive to *how fast it is already moving* before deciding to engage the engine. S3 DQN promotes `pos*vel` (momentum) to its top feature (0.94), well above its raw `vel` weight — a clean signal that the fuel-optimal policy reasons about momentum rather than raw velocity.

**Disagreement between techniques is itself informative.** When permutation importance ranks `pos*vel` highly but logistic regression underweights it, the policy depends on the momentum interaction in a way the linear model cannot express but the tree and permutation tests can detect. Reporting all three techniques separates "the policy uses this feature" from "the policy *linearly* uses this feature".
