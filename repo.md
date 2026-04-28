# RL-Group — Mountain Car Repository Guide

A walkthrough of this repository, the four scenarios it implements, and the
shared `mclib/` library that backs them. Written for someone joining the
project (e.g. to add the **Explanation Tools** section) and needing to know
*what is where* and *which file to touch for what*.

## 1. Project goal

Train and analyze RL agents on the classic Mountain Car environment under
**four scenarios** that vary along two axes:

|        | Discrete actions | Continuous actions |
|--------|------------------|--------------------|
| Min steps  | **S1**           | **S2**             |
| Min fuel / action-use | **S3**           | **S4**             |

Each scenario uses its own reward design (a `gym.Wrapper`) but is evaluated
on a common objective so results are comparable. The deliverables are:

1. Per-scenario analysis (curves, policy heatmaps, value surfaces, phase
   portraits, statistical metrics).
2. Cross-scenario comparison.
3. **Explanation Tools** — fit a multinomial logistic regression on each
   learned policy to quantify which state features the agent relies on.

## 2. Top-level layout

```
RL-Group/
├── RLI_22_A0_MountainCar.ipynb     # The single master notebook (all scenarios)
├── info.md                         # Assignment brief
├── notes.md                        # S4 retuning / debugging log
├── pyproject.toml                  # Deps: gymnasium, SB3, torch, tensorboard
├── mclib/                          # Shared library — see §4
├── models/                         # Saved agent checkpoints
│   ├── s1/                         # tabular .npz, DQN .pt, eval .npy traces
│   ├── s3/                         # same layout, fuel variant
│   └── scenario4/                  # 8 SB3 .zip files (SAC + TD3 × 4 rewards)
└── runs/                           # TensorBoard event files (16 runs)
```

Everything user-facing happens in **`RLI_22_A0_MountainCar.ipynb`**. The
notebook imports `mclib`; almost no logic should live in the notebook
itself.

## 3. The four scenarios

### Scenario 1 — Discrete, minimum steps  ✅ done
Standard `MountainCar-v0`. Reward `-1` per step, `0` on goal (Gymnasium
default). Goal: reach the flag in as few steps as possible.

- Notebook cells: ~8–40
- Env factories: `mc.make_s1()` (default reward), `mc.make_s1_shaped()`
  (potential-based shaping via `EnergyShapingWrapper`)
- Agents trained: tabular **Q-learning**, **SARSA**, **DQN**
- Saved artifacts: `models/s1/{ql,sarsa,dqn}*` (rewards, eval means/stds,
  final policies, model weights)
- TensorBoard: `runs/S1_QL_default`, `runs/S1_QL_shaped`, `runs/S1_SARSA`,
  `runs/S1_DQN`

### Scenario 2 — Continuous, minimum steps  ❌ not implemented
Continuous action space `MountainCarContinuous-v0` with a reward that
penalizes time but **not** action magnitude — i.e. the continuous analogue
of S1.

- Notebook cell 41 is an empty placeholder.
- The factory `mc.make_s2()` already exists in `mclib/envs/wrappers.py`.
  Inspect it before writing S2: it is the intended baseline reward for this
  scenario.
- No models exist under `models/` for S2; no `runs/S2_*` TensorBoard logs.
- Suggested approach when filling this in: mirror the S4 pipeline (SAC and
  TD3 from SB3, same training/eval helpers in `mclib/training/continuous.py`)
  but swap the wrapper to `make_s2()`.

### Scenario 3 — Discrete, minimum fuel  ✅ done
Same discrete action space as S1, but the reward design rewards reaching
the goal *with as little engine use as possible*.

- Notebook cells: ~43–77
- Wrappers: `DiscreteFuelWrapper` (V1, used by `make_s3()`),
  `DiscreteFuelWrapperV2` (alternative shaping, used by `make_s3_v2()`)
- Agents trained: tabular Q-learning (V1 and V2), SARSA, DQN
- Saved artifacts: `models/s3/{ql_fuel,ql_fuel_v2,sarsa_fuel,dqn_fuel}*`
- TensorBoard: `runs/S3_QL_fuel`, `runs/S3_V2_QL_fuel`, `runs/S3_SARSA_fuel`,
  `runs/S3_DQN_fuel`

### Scenario 4 — Continuous, minimum action-use  ✅ done (with caveats)
Continuous action space, reward penalizes any non-null thrust. See
`notes.md` for the full debugging story — the pure action-use reward
creates a strong "do nothing" local optimum, so four reward variants are
trained side by side:

| Variant         | Factory                    | Purpose                                    |
|-----------------|----------------------------|--------------------------------------------|
| `S4_step`       | `mc.make_s4()`             | Min-time baseline, sanity check on dynamics |
| `S4_default`    | `mc.make_s4_default()`     | Gymnasium default `-0.1·a² + 100`          |
| `S4_action_use` | `mc.make_s4_action_use()`  | The "true" S4 reward                        |
| `S4_shaped`     | `mc.make_s4_shaped()`      | Action-use + progress + velocity bonus     |

- Notebook cells: ~78–99
- Agents: **SAC** and **TD3** (Stable-Baselines3), one of each per reward
  variant → 8 trained models.
- Saved artifacts: `models/scenario4/S4_{variant}_{SAC,TD3}.zip`
- TensorBoard: `runs/S4_{variant}_{SAC,TD3}_1` (8 runs)
- Final evaluation always runs under `make_s4_action_use()` (the pure S4
  reward), so all 8 agents are scored on the same objective.

## 4. The `mclib/` library

Every reusable piece of logic lives in `mclib/`. Treat the notebook as a
thin orchestration layer over these modules.

```
mclib/
├── __init__.py              # Public API: re-exports everything below
├── agents/
│   ├── tabular.py           # TabularQLearningAgent, SarsaAgent (40×40 bins)
│   ├── dqn.py               # QNetwork, DQNAgent (replay + target net)
│   ├── sac.py               # SACAgent — wraps SB3 SAC
│   └── td3.py               # TD3Agent — wraps SB3 TD3
├── envs/
│   └── wrappers.py          # All reward wrappers + make_s{1,2,3,4}* factories
├── training/
│   ├── loops.py             # train_tabular / train_sarsa / train_dqn + evals
│   └── continuous.py        # SB3 model factory, eval callback, S4 hyperparams
├── visualization/
│   └── plots.py             # heatmaps, value surfaces, phase portraits, etc.
├── analysis/
│   └── explain.py           # ⭐ multinomial logistic regression on policies
└── testbed/
    └── runner.py            # Testbed class — TB writer + result dict
```

### What lives where, in detail

- **`mclib/agents/tabular.py`** — `TabularQLearningAgent`, `SarsaAgent`.
  Discretizes `(pos, vel)` into a 40×40 grid. Exposes `get_policy_grid()`,
  `get_value_grid()`, `get_visit_grid()` — these are the 40×40 numpy
  arrays you feed to `explain_policy_regression`.
- **`mclib/agents/dqn.py`** — `DQNAgent` and `QNetwork`. Same
  `get_q_values_grid()` interface returns the policy + value grids
  evaluated on the same 40×40 discretization.
- **`mclib/agents/sac.py`, `td3.py`** — SB3 wrappers. Continuous policies,
  so the "policy" is a real-valued action grid. Use
  `mclib.visualization.plots.get_continuous_policy_grid(model, n_bins=40)`
  to extract it.
- **`mclib/envs/wrappers.py`** — all reward shaping classes
  (`DiscreteFuelWrapper`, `DiscreteFuelWrapperV2`, `ContinuousStepsWrapper`,
  `ContinuousActionUseWrapper`, `ContinuousShapedRewardWrapper`,
  `ContinuousLinearActionWrapper`, `EnergyShapingWrapper`) plus the
  factories `make_s1`, `make_s1_shaped`, `make_s2`, `make_s3`, `make_s3_v2`,
  `make_s4`, `make_s4_default`, `make_s4_action_use`, `make_s4_shaped`,
  `make_s4_linear_action`.
- **`mclib/training/loops.py`** — discrete training loops with TensorBoard
  logging (`episode_reward`, `epsilon`, `eval_mean`, `eval_std`).
- **`mclib/training/continuous.py`** — SB3 factory
  `make_sb3_continuous_model`, `train_sb3_continuous`, the
  `ContinuousEvalCallback` (logs success rate, mean steps, fuel, linear
  effort to TB), plus S4 constants (`S4_DEFAULT_SEED=42`,
  `S4_TOTAL_TIMESTEPS=150_000`, `S4_TD3_TOTAL_TIMESTEPS=300_000`,
  `S4_MAX_STEPS=999`).
- **`mclib/visualization/plots.py`** — `plot_training_curve`,
  `plot_policy_heatmap`, `plot_continuous_action_heatmap`,
  `plot_value_surface_3d`, `plot_phase_portrait`, `collect_trajectories`,
  `count_steps`, `get_continuous_policy_grid`, `get_sac_policy_grid`.
  Constants `ACTION_CMAP`, `ACTION_COLORS`, `ACTION_LABELS` for consistent
  colors across scenarios.
- **`mclib/testbed/runner.py`** — `Testbed` class. Owns a
  `SummaryWriter`, runs an agent, stores `rewards`, `eval_means`,
  `eval_stds`, `final` keyed by run name. Useful when adding new runs.
- **`mclib/analysis/explain.py`** — ⭐ the file most relevant to the
  Explanation Tools section. Contains:
  - `multinomial_logreg(X, y, n_classes=3, lr=0.05, n_iters=600, lam=0.01)`
    — pure-NumPy multinomial logistic regression with L2.
  - `explain_policy_regression(policy_grid, n_bins=40, scenario_name=...)`
    — fits the regression on the engineered feature set
    `[pos, vel, pos², vel², pos·vel, |vel|, sin(3·pos)]`, prints the
    accuracy, and returns ranked feature importances. **This is the
    starting point for your explanation work.** The current implementation
    is hard-coded for 3 discrete actions and prints to stdout — extending
    it for the continuous scenarios (S2, S4) will require either binning
    the action into a sign/magnitude class or switching to a regression
    target.

## 5. Saved artifacts cheat sheet

`models/s1/` and `models/s3/` follow the same convention:

- `{agent}_rewards.npy`         — per-episode training reward
- `{agent}_eval_means.npy`      — eval mean reward over time
- `{agent}_eval_stds.npy`       — eval std over time
- `{agent}_eval_eps.npy`        — episode indices at which eval was run
- `{agent}_eval_steps.npy`      — eval mean episode length (DQN only)
- `{agent}_final.npy`           — final 100-episode evaluation summary
- `{agent}.npz` / `{agent}.pt`  — model weights (npz = tabular Q-table,
                                  pt = PyTorch DQN)

`models/scenario4/` stores SB3 checkpoints as `.zip` files:
`S4_{step,default,action_use,shaped}_{SAC,TD3}.zip`. Load them with
`load_sb3_continuous_model` (see `mclib/training/continuous.py`).

## 6. TensorBoard

Logs live in `runs/`. Launch with:

```bash
tensorboard --logdir runs
```

The 16 existing runs cover S1 (4), S3 (4), S4 (8). No S2 runs yet.

## 7. Where the Explanation Tools work plugs in

The user's task is the third deliverable in `info.md`. The bridge is:

1. For each trained agent, get a 40×40 policy grid.
   - Discrete (S1, S3): `agent.get_policy_grid()` already returns this.
   - Continuous (S2, S4): use `get_continuous_policy_grid(model)` from
     `mclib/visualization/plots.py`. Decide upfront whether to bin
     continuous actions into classes for the multinomial regression or to
     switch to a real-valued regression.
2. Feed the grid into `explain_policy_regression` (or an extended version
   of it) — currently in `mclib/analysis/explain.py`.
3. Compare feature importances across scenarios — this is the comparative
   bit in §3 of the *Explanation Tools* part of `info.md`.
4. Visualize: feature-importance bar charts per scenario, and overlay the
   regression decision boundary on the existing policy heatmaps.

## 8. Open issues / things to watch out for

- **S2 is missing.** Either implement it (mirror S4 with `make_s2()`) or
  document the gap in the report. The Explanation Tools comparison loses
  the `S1 vs S2` discrete-vs-continuous min-steps axis without it.
- **`explain_policy_regression` is discrete-only.** Extending it to S4's
  continuous policies needs a design choice (action binning vs. regression
  target).
- **S4 action-use agents may have collapsed to ~zero force** (per
  `notes.md`). If the policy grid is uniformly idle, the regression will
  show ~zero feature importance — that's a real finding, not a bug.
- **Reward design vs. evaluation reward.** S4 evaluates all variants under
  the pure action-use reward. When comparing scenarios in the Explanation
  Tools section, make sure to be explicit about whether feature importances
  are computed on the *training* policy or after re-evaluation.
