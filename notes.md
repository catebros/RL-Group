# Scenario 4 Training Notes

## Latest Diagnosis

The implementation was checked against the saved notebook outputs, TensorBoard event logs, action heatmaps, and phase portraits. The wrappers and evaluation plumbing were behaving as intended. The main failure was caused by reward incentives, not by a code path bug.

The pure action-use reward was:

```text
reward = -0.01
reward -= 0.10 * float(abs(action) > 1e-3)
reward -= 0.02 * abs(action)
reward += 100.0 on goal
```

This is conceptually aligned with Scenario 4 because it directly penalizes non-null engine-use events. However, it creates a strong no-action local optimum before the goal is discovered:

```text
failed no-op episode:              about -9.99
failed exploratory active episode: about -110 to -130
```

As a result, both action-use agents learned almost exactly zero force. The notebook output showed `0%` success, `999` steps, `0.00` mean non-null actions, and mean absolute action around `0.0003` to `0.0006`. The policy heatmaps were flat near zero, and the regression explainer had zero feature importances.

TD3's minimum-time failure was separate. `S4_step_TD3` was not idle: it used large actions, but the phase portrait showed a tight valley orbit rather than the large left-right momentum-building trajectory needed to reach the flag. This is an exploration and credit-assignment failure for deterministic TD3 in a deceptive sparse-reward task.

## Adjusted Approach Applied

Scenario 4 now uses a reward hierarchy instead of treating the pure action-use reward as the only alternative:

| Setup | Factory | Reward role |
|---|---|---|
| A | `mc.make_s4()` | Minimum-time baseline: `-1 + 100` on goal |
| B | `mc.make_s4_default()` | Standard continuous baseline: Gymnasium default `-0.1 * action^2 + 100` on goal |
| C | `mc.make_s4_action_use()` | Pure Scenario 4 non-null action-use objective |
| D | `mc.make_s4_shaped()` | Shaped action-use training reward with progress and velocity feedback |

The shaped reward keeps the Scenario 4 action-use terms and adds:

```text
reward += 1.0 * (x_next - x)
reward += 0.1 * abs(v_next)
```

This should be described as a training aid, not as the pure Scenario 4 objective. All final notebook summaries now evaluate every trained policy under the pure action-use environment, so the comparison is based on the same objective metrics.

## TD3 Changes

TD3 was updated to match the common Stable-Baselines3/RL-Zoo style for `MountainCarContinuous-v0` more closely:

```text
total_timesteps = 300000
action_noise = OrnsteinUhlenbeckActionNoise(sigma=0.5)
learning_rate = 1e-3
buffer_size = 1000000
batch_size = 64
gamma = 0.99
tau = 0.005
policy_delay = 2
train_freq = (1, "step")
gradient_steps = 1
```

The longer budget and OU noise are intended to improve TD3's exploration. They do not guarantee that TD3 will solve the pure action-use reward, because that reward still makes no-action behavior attractive until the goal is discovered.

## Report Interpretation

Use the pure action-use result as an important reward-design finding if it still fails: a scenario-correct objective can be difficult to learn from scratch when the terminal reward is sparse and exploration actions are immediately penalized.

The strongest report structure is:

1. Minimum-time baseline shows whether the algorithms can solve the dynamics.
2. Default continuous baseline connects the experiment to the common Gymnasium setup.
3. Pure action-use reward tests the exact Scenario 4 objective.
4. Shaped action-use reward tests whether engineered feedback can overcome the no-action local optimum.

Final performance should be compared with:

- success rate
- mean steps
- mean non-null actions
- mean absolute action / linear effort
- fuel `sum(action^2)`
- return under the pure Scenario 4 action-use reward
- action heatmaps and phase portraits

If the shaped reward still fails, the likely reason is that the progress and velocity bonuses are small compared with the non-null action penalty. That should be reported as evidence that the Scenario 4 objective is especially sensitive to reward scale.
