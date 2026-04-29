# Explanation Tools — Report Figures

This isolates the figures and tables that carry the explanation-tool conclusions for each scenario and for the cross-scenario synthesis. All figures are produced by the master notebook `RLI_22_A0_MountainCar.ipynb`; cell IDs are referenced so the originals are easy to relocate.

The shared engineered feature set behind every figure is:

```
[pos, vel, pos^2, vel^2, pos*vel, |vel|, sin(3*pos)]
```

Continuous-action grids (S2, S4) are binned into `{0=left, 1=idle, 2=right}` (deadband 0.05) so all 14 policies share a 3-class output space.

---

## Figure 1 — Cross-scenario feature importance (3-panel bar chart)

**Where:** Section 7.1 of the notebook, cell `explain-bar-charts`. Built by `mc.plot_feature_importance_bars(all_results, technique=...)` for `technique ∈ {logreg, tree, permutation}`.

**What it shows:** One row per technique, x-axis = the seven engineered features, bars grouped by the 14 trained policies (S1 QL/SARSA/DQN, S3 QL/SARSA/DQN, S4 step/default/action_use/shaped × SAC/TD3). The y-axis label adapts to the technique — `mean |coef|` for logreg, impurity importance for the tree, accuracy drop for permutation.

**Why it matters (this is the headline figure of the section):**

- **Velocity dominance is universal.** `vel` is the tallest bar in nearly every solvable policy across all three panels — the agent's first-order rule is "thrust in the direction of motion to build momentum." For TD3 runs `vel`'s logreg coefficient sits in the 1.15–1.47 range, well above any other feature.
- **The S1 → S3 reward shift is visible.** S3 DQN promotes `pos*vel` (the momentum interaction) to its top feature (0.94, ahead of raw `vel` at 0.54). Tabular S3 agents lift `vel^2` and `|vel|` above their S1 counterparts. This is the cleanest quantitative signal that the fuel reward changed *what the policy reasons about*, not just where it draws the boundary.
- **The S4 action_use_SAC collapse is unmissable.** All seven of its logreg bars sit at 0.000. That is the "constant policy" signature — the SAC agent learned to output ~zero force everywhere under the pure action-use reward, so no feature can linearly separate the (essentially single) class.
- **Logreg vs tree vs permutation disagreements are diagnostic.** Where the tree/permutation panels rank `pos*vel` or `pos^2` higher than logreg does, the policy is using that feature *non-linearly*. Reporting all three is what separates "the policy uses this feature" from "the policy *linearly* uses this feature."

---

## Figure 2 — Cross-scenario logreg decision boundary overlays (5×3 grid)

**Where:** Section 7.1 of the notebook, cell `explain-boundary-overlays`. Built by `mc.plot_decision_boundary_overlay(policy_grid, predictions_grid, ...)` for each of the 14 policies.

**What it shows:** Each panel plots the binned policy heatmap (`pos` on x, `vel` on y) with a black `x` on every `(pos, vel)` cell where the multinomial-logreg surrogate predicts a different action than the true policy. Empty panels = surrogate matches everywhere.

**Why it matters:**

- **Locates the non-linear structure in state space.** For TD3 variants the disagreement marks cluster on a thin diagonal seam — the boundary between left and right thrust — and the surrogate is otherwise correct. That seam is exactly the velocity-zero crossing, which a single linear cut struggles to draw cleanly.
- **Confirms the SAC coasting region is non-linear.** S2 and S4 SAC variants show a horizontal band of disagreement around `vel ≈ 0`: the true policy idles there; the linear surrogate forces a left/right choice. This is the geometric reason their logreg accuracy (73–82%) is lower than TD3's (~94%) — the idle band cannot be expressed as a hyperplane.
- **Shows the action_use_SAC collapse visually.** The panel is essentially solid black `x`s — the surrogate disagrees almost everywhere because there is no learnable structure to recover.
- **S1/S3 tabular agents show scattered, per-cell disagreements.** That scatter is the visual fingerprint of a tabular policy: each grid cell makes its own decision, so the linear model misses the per-cell switching even when the overall pattern looks structured.

---

## Figure 3 — Cross-scenario explainability accuracy table

**Where:** Section 7.1 of the notebook, cell `explain-run` (text output). Reproduced verbatim from the notebook:

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

**Why it matters (this is the numerical anchor for the whole section):**

- **A single number per agent saying "how explainable is this policy by simple rules."** Logreg accuracy is bounded by linearity; tree accuracy at depth 4 is bounded by axis-aligned rules; permutation baseline equals logreg by construction (same scorer).
- **TD3 is consistently the most explainable family** (logreg 92–95%, tree 94–97%) — every TD3 variant converges to a near-bang-bang controller dominated by `vel`, which a linear model captures cleanly.
- **DQN sits in a middle band** (S1 87.2%, S3 82.9% logreg) — the neural value function smooths out the per-cell switching that hurts the tabular agents.
- **Tabular S1/S3 cluster at ~55–60% logreg.** This is *not* a sign of a bad policy; it is the signature of a tabular agent making per-cell decisions that no 7-feature linear boundary can mimic. The tree closes the gap to ~73%, confirming the structure exists but is non-linear.
- **`S4 action_use_SAC`: 43.5% logreg, 100% tree.** This pair is the textbook diagnostic of a *constant policy* — logreg can't beat majority class because there is no signal, and the tree fits the constant trivially with one node. The figures in §1 and §2 say the same thing visually; this row says it numerically.

---

## Figure 4 — S2 policy vs logreg surrogate (2-panel side-by-side)

**Where:** Section 3 of the notebook (Scenario 2), cell `0e8d6238`. Generated alongside the per-scenario S2 feature importance bars.

**What it shows:** Two policy heatmaps placed side by side for the first S2 run: (left) the binned true policy, (right) the policy predicted by re-applying the fitted logreg weights to the same `(pos, vel)` grid.

**Why it matters as an isolated illustration:**

- **It makes "what the surrogate misses" visually obvious.** The right panel is a clean two-region map split by a near-vertical seam at `vel = 0`. The left panel shows the same seam but with scattered cyan idle cells around the velocity-zero band — the SAC coasting region. The linear surrogate cannot reproduce idle, so it overwrites it with the nearest of left/right.
- **It is the per-scenario companion to Figure 2.** Where Figure 2's overlay shows *where* the surrogate fails as black `x`s, Figure 4 shows the actual surrogate prediction grid — the two panels together make the failure mode interpretable rather than just locating it.
- **Pair this with the S2 logreg numbers** (default SAC 79.2%, shaped SAC 73.7%, default TD3 94.4%, shaped TD3 95.6%) to anchor the qualitative read in the accuracies.

---

## How these four pieces fit together in the report

- **Lead with Figure 1.** It carries the cross-scenario story in one image: velocity dominance, the S1→S3 fuel-reward shift toward `pos*vel`/`|vel|`, and the action_use_SAC collapse all read directly off the bars.
- **Use Figure 3 as the numerical anchor.** The table converts the visual story into single numbers per agent and gives the reader the "how explainable is each policy" axis.
- **Use Figure 2 to localize.** Once the reader knows *which* policies are non-linear (from Figure 3), Figure 2 shows *where* in `(pos, vel)` space they are non-linear — concentrated on the velocity-zero seam for SAC, scattered per-cell for tabular agents, everywhere for the collapsed action_use_SAC.
- **Use Figure 4 as a worked example.** Pull one SAC panel out of Figure 2 and pair it with its surrogate prediction so the failure mode of the linear model is unambiguous to a non-technical reader.

The remaining per-scenario explainer outputs (S2/S3/S4 feature importance bars, individual logreg printouts) are subsumed by Figures 1 and 3 and are best cited rather than reproduced.
