import os

import numpy as np


S4_DEFAULT_SEED = 42
S4_TOTAL_TIMESTEPS = 150_000
S4_TD3_TOTAL_TIMESTEPS = 300_000
S4_MAX_STEPS = 999
TD3_ACTION_NOISE_SIGMA = 0.5
TD3_ACTION_NOISE_TYPE = "ornstein_uhlenbeck"
TD3_GAMMA = 0.99
TD3_LEARNING_STARTS = 10_000

SB3_COMMON_KWARGS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 256,
    "learning_starts": 1_000,
    "tau": 0.005,
}

SB3_ALGO_KWARGS = {
    "SAC": {
        "buffer_size": 50_000,
        "batch_size": 512,
        "learning_starts": 0,
        "gamma": 0.9999,
        "tau": 0.01,
        "train_freq": 32,
        "gradient_steps": 32,
        "ent_coef": 0.1,
        "use_sde": True,
        "policy_kwargs": {"log_std_init": -3.67, "net_arch": [64, 64]},
    },
    "TD3": {
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "batch_size": 64,
        "gamma": TD3_GAMMA,
        "learning_starts": TD3_LEARNING_STARTS,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
    },
}


def make_sb3_continuous_model(
    algorithm,
    env_factory,
    seed=S4_DEFAULT_SEED,
    tensorboard_log="runs",
    policy="MlpPolicy",
    verbose=0,
    **kwargs,
):
    """Create a Stable-Baselines3 continuous-control model with shared defaults."""
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    algorithms = {"SAC": SAC, "TD3": TD3}
    algo_name = algorithm.upper() if isinstance(algorithm, str) else algorithm.__name__.upper()
    if algo_name not in algorithms:
        raise ValueError(f"Unsupported continuous algorithm: {algorithm!r}")

    env = env_factory()
    params = {
        **SB3_COMMON_KWARGS,
        **SB3_ALGO_KWARGS[algo_name],
        **kwargs,
        "policy": policy,
        "env": env,
        "seed": seed,
        "tensorboard_log": tensorboard_log,
        "verbose": verbose,
    }
    if algo_name == "TD3" and "action_noise" not in params:
        action_dim = int(np.prod(env.action_space.shape))
        noise_cls = (
            OrnsteinUhlenbeckActionNoise
            if TD3_ACTION_NOISE_TYPE == "ornstein_uhlenbeck"
            else NormalActionNoise
        )
        params["action_noise"] = noise_cls(
            mean=np.zeros(action_dim),
            sigma=TD3_ACTION_NOISE_SIGMA * np.ones(action_dim),
        )
    return algorithms[algo_name](**params)


def load_sb3_continuous_model(
    algorithm,
    model_path,
    env_factory,
    seed=S4_DEFAULT_SEED,
    verbose=0,
):
    """Load a saved Stable-Baselines3 continuous-control model with a fresh env."""
    from stable_baselines3 import SAC, TD3

    algorithms = {"SAC": SAC, "TD3": TD3}
    algo_name = algorithm.upper() if isinstance(algorithm, str) else algorithm.__name__.upper()
    if algo_name not in algorithms:
        raise ValueError(f"Unsupported continuous algorithm: {algorithm!r}")

    env = env_factory()
    return algorithms[algo_name].load(
        model_path,
        env=env,
        seed=seed,
        verbose=verbose,
    )


def _make_eval_callback(eval_env_factory, eval_freq, n_eval_episodes, deterministic):
    from stable_baselines3.common.callbacks import BaseCallback

    class ContinuousEvalCallback(BaseCallback):
        """Lightweight evaluation callback that records S4 metrics to TensorBoard."""

        def __init__(self):
            super().__init__()
            self.eval_timesteps = []
            self.eval_mean_rewards = []
            self.eval_std_rewards = []
            self.eval_success_rates = []
            self.eval_mean_steps = []

        def _on_step(self):
            if self.n_calls % eval_freq != 0:
                return True

            result = evaluate_continuous_policy(
                self.model,
                eval_env_factory,
                n_episodes=n_eval_episodes,
                deterministic=deterministic,
            )
            summary = result["summary"]
            self.eval_timesteps.append(self.num_timesteps)
            self.eval_mean_rewards.append(summary["mean_reward"])
            self.eval_std_rewards.append(summary["std_reward"])
            self.eval_success_rates.append(summary["success_rate"])
            self.eval_mean_steps.append(summary["mean_steps"])

            self.logger.record("s4_eval/mean_reward", summary["mean_reward"])
            self.logger.record("s4_eval/std_reward", summary["std_reward"])
            self.logger.record("s4_eval/success_rate", summary["success_rate"])
            self.logger.record("s4_eval/mean_steps", summary["mean_steps"])
            self.logger.record("s4_eval/mean_fuel", summary["mean_fuel"])
            self.logger.record("s4_eval/mean_linear_effort", summary["mean_linear_effort"])
            self.logger.dump(self.num_timesteps)
            return True

    return ContinuousEvalCallback()


def train_sb3_continuous(
    algorithm,
    env_factory,
    run_name,
    total_timesteps=S4_TOTAL_TIMESTEPS,
    seed=S4_DEFAULT_SEED,
    tensorboard_log="runs",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic_eval=True,
    verbose=0,
    model_save_path=None,
    **model_kwargs,
):
    """
    Train a SAC/TD3 model on a continuous-control environment.

    Returns a dictionary containing the trained model and evaluation traces.
    """
    model = make_sb3_continuous_model(
        algorithm,
        env_factory,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        **model_kwargs,
    )
    callback = _make_eval_callback(
        env_factory,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic_eval,
    )
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=run_name,
        callback=callback,
        progress_bar=False,
    )
    if model_save_path is not None:
        model.save(model_save_path)
    return {
        "model": model,
        "run_name": run_name,
        "log_dir": os.path.join(tensorboard_log, run_name),
        "eval_timesteps": callback.eval_timesteps,
        "eval_mean_rewards": callback.eval_mean_rewards,
        "eval_std_rewards": callback.eval_std_rewards,
        "eval_success_rates": callback.eval_success_rates,
        "eval_mean_steps": callback.eval_mean_steps,
    }


def evaluate_continuous_policy(
    model,
    env_factory,
    n_episodes=100,
    max_steps=S4_MAX_STEPS,
    deterministic=True,
    seed=7_000,
    non_null_threshold=1e-3,
):
    """
    Evaluate a continuous-control policy and return raw episode metrics plus a summary.

    Success is based on environment termination, not reward thresholds.
    """
    episodes = []
    env = env_factory()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        fuel = 0.0
        linear_effort = 0.0
        non_null_actions = 0
        max_abs_action = 0.0
        success = False
        steps = max_steps

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=float).reshape(-1)
            applied_action = float(np.clip(action_arr[0], -1.0, 1.0))

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            fuel += applied_action ** 2
            linear_effort += abs(applied_action)
            max_abs_action = max(max_abs_action, abs(applied_action))
            non_null_actions += int(abs(applied_action) > non_null_threshold)

            if terminated or truncated:
                success = bool(terminated)
                steps = step + 1
                break

        episodes.append(
            {
                "episode": ep,
                "reward": total_reward,
                "success": success,
                "steps": steps,
                "fuel": fuel,
                "linear_effort": linear_effort,
                "non_null_actions": non_null_actions,
                "mean_abs_action": linear_effort / steps if steps else 0.0,
                "max_abs_action": max_abs_action,
            }
        )

    env.close()
    return {"episodes": episodes, "summary": summarize_continuous_metrics(episodes)}


def summarize_continuous_metrics(episodes):
    """Summarize episode metrics with means, standard deviations, and 95% CIs."""
    n = len(episodes)
    rewards = np.array([ep["reward"] for ep in episodes], dtype=float)
    successes = np.array([ep["success"] for ep in episodes], dtype=bool)
    steps = np.array([ep["steps"] for ep in episodes], dtype=float)
    fuel = np.array([ep["fuel"] for ep in episodes], dtype=float)
    linear_effort = np.array([ep["linear_effort"] for ep in episodes], dtype=float)
    non_null = np.array([ep["non_null_actions"] for ep in episodes], dtype=float)
    mean_abs = np.array([ep["mean_abs_action"] for ep in episodes], dtype=float)
    max_abs = np.array([ep["max_abs_action"] for ep in episodes], dtype=float)

    def std(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    def ci95(x):
        return 1.96 * std(x) / np.sqrt(len(x)) if len(x) else 0.0

    successful_steps = steps[successes]
    return {
        "n_episodes": n,
        "mean_reward": float(np.mean(rewards)) if n else 0.0,
        "std_reward": std(rewards),
        "reward_ci95": ci95(rewards),
        "success_rate": float(np.mean(successes)) if n else 0.0,
        "mean_steps": float(np.mean(steps)) if n else 0.0,
        "std_steps": std(steps),
        "steps_ci95": ci95(steps),
        "mean_steps_success": float(np.mean(successful_steps)) if len(successful_steps) else np.nan,
        "mean_fuel": float(np.mean(fuel)) if n else 0.0,
        "std_fuel": std(fuel),
        "mean_linear_effort": float(np.mean(linear_effort)) if n else 0.0,
        "mean_non_null_actions": float(np.mean(non_null)) if n else 0.0,
        "mean_abs_action": float(np.mean(mean_abs)) if n else 0.0,
        "mean_max_abs_action": float(np.mean(max_abs)) if n else 0.0,
    }
