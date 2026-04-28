# Explanation Tools — Evaluation Plan

How to implement the **Explanation Tools** deliverable from `info.md` while
keeping the existing `mclib/` repository structure.

## 1. Where the code goes

Following the existing structure, everything explanation-related belongs in
`mclib/analysis/`. That package currently has just `explain.py`; the
convention from sibling packages like `mclib/agents/` (4 files) is **one
file per technique**. A clean layout:

```
mclib/analysis/
├── explain.py          # already exists — logistic regression
├── tree.py             # NEW — decision tree surrogate
└── importance.py       # NEW — permutation / behavioral importance
```

Re-export the new public functions from `mclib/__init__.py` (next to the
existing `explain_policy_regression`). The notebook stays a thin
orchestrator: a single new section that loads each scenario's policy grid,
runs the three analyses, and plots them — no logic in the notebook itself.

For getting policy grids in a uniform shape across scenarios, lean on what
already exists:

- `agent.get_policy_grid()` for tabular / DQN (S1, S3)
- `get_continuous_policy_grid(model)` from
  `mclib/visualization/plots.py` for SAC / TD3 (S4)

For the continuous case (S4 — and S2 if implemented), bin the signed action
into 3 classes (`left / idle / right` via `np.sign` with a deadband around
zero) so all three approaches can run uniformly across scenarios. This also
gives a directly-comparable axis for the cross-scenario comparison
`info.md` asks for.

## 2. Three approaches, different aspects

Each technique answers a different question about the learned policy.

| # | Approach | Question it answers | Strength | Weakness |
|---|----------|---------------------|----------|----------|
| 1 | **Multinomial logistic regression** (extend `explain.py`) | *Which engineered features linearly separate the actions?* | Per-feature, per-action coefficients; already half-built | Only linear — accuracy itself measures "how non-linear is this policy" |
| 2 | **Decision tree surrogate** (new `tree.py`, sklearn `DecisionTreeClassifier`) | *What if-then rules approximate the policy?* | Captures non-linearity; readable rules ("if vel > 0.02 → push right"); tree depth / leaf count quantify policy complexity | Can be unstable; cap depth (3–5) for interpretability |
| 3 | **Permutation feature importance** (new `importance.py`, model-agnostic) | *Which features does the agent actually rely on, regardless of how we model it?* | No surrogate model assumed; identical on raw `(pos, vel)` and engineered features; comparable across discrete and continuous scenarios | Computationally heavier; depends on a baseline accuracy score |

The three angles together cover:

1. **Statistical / linear** view — logistic regression coefficients
2. **Rule-based / hierarchical** view — decision-tree if-then structure
3. **Behavioral / sensitivity** view — permutation importance

They should mostly agree on dominant features (likely `vel` and
`sin(3·pos)`) but disagree on secondary ones — that disagreement is itself
a reportable finding.

## 3. Minimum / fallback subsets

- **If dropping one**: keep logistic + permutation. Permutation is the most
  independent from logistic regression and adds the model-agnostic angle.
- **If keeping all three**: the decision tree adds the "natural"
  interpretation, since Mountain Car policies are genuinely rule-shaped
  (terrain + velocity → action).

## 4. Suggested deliverable structure (per `info.md`)

For each of the three techniques:

1. **Numbers** — accuracy / R² / importance score per scenario (S1, S3, S4
   at minimum; S2 if implemented).
2. **Comparison across scenarios** — does S3 (min fuel) lean on `|vel|`
   more than S1 (min steps)? Does S4 rely on `sin(3·pos)` differently?
3. **Visualization** —
   - Feature-importance bar charts (one panel per scenario, side by side).
   - Decision-boundary / tree-partition overlay on the existing policy
     heatmap (`plot_policy_heatmap`) to show where the surrogate agrees
     and disagrees with the true policy.
4. **Physical interpretation** — translate the dominant features into
   physics ("agent thrusts when slow, idles when fast" → velocity-driven
   policy).

## 5. Open question

Three separate files (`explain.py`, `tree.py`, `importance.py`) keeps each
technique's API focused and matches the per-technique convention used in
`mclib/agents/`. The alternative is to merge everything into
`explain.py` if the team prefers one entry point — both are reasonable.
