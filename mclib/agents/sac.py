import os

from ..training.continuous import (
    S4_DEFAULT_SEED,
    S4_TOTAL_TIMESTEPS,
    S4_MAX_STEPS,
    evaluate_continuous_policy,
    load_sb3_continuous_model,
    make_continuous_eval_callback,
    make_sb3_continuous_model,
)


class SACAgent:
    """Stable-Baselines3 SAC wrapper with the same outer shape as mclib agents."""

    algorithm = "SAC"

    def __init__(
        self,
        env_factory=None,
        seed=S4_DEFAULT_SEED,
        tensorboard_log="runs",
        policy="MlpPolicy",
        verbose=0,
        model=None,
        **model_kwargs,
    ):
        self.env_factory = env_factory
        self.seed = seed
        self.tensorboard_log = tensorboard_log
        self.policy = policy
        self.verbose = verbose
        self.model_kwargs = dict(model_kwargs)
        self.model = model

        if self.model is None and self.env_factory is not None:
            self.build()

    def build(self):
        if self.env_factory is None:
            raise ValueError("SACAgent requires an env_factory before it can build a model.")

        self.model = make_sb3_continuous_model(
            self.algorithm,
            self.env_factory,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            policy=self.policy,
            verbose=self.verbose,
            **self.model_kwargs,
        )
        return self.model

    def train(
        self,
        run_name,
        total_timesteps=S4_TOTAL_TIMESTEPS,
        eval_env_factory=None,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic_eval=True,
        model_save_path=None,
        progress_bar=False,
    ):
        if self.model is None:
            self.build()

        eval_factory = eval_env_factory or self.env_factory
        callback = make_continuous_eval_callback(
            eval_factory,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic_eval,
        )
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_name,
            callback=callback,
            progress_bar=progress_bar,
        )
        if model_save_path is not None:
            self.save(model_save_path)

        return {
            "agent": self,
            "model": self.model,
            "run_name": run_name,
            "log_dir": os.path.join(self.tensorboard_log, run_name),
            "model_path": str(model_save_path) if model_save_path is not None else None,
            "rewards": [],
            "eval_timesteps": callback.eval_timesteps,
            "eval_mean_rewards": callback.eval_mean_rewards,
            "eval_std_rewards": callback.eval_std_rewards,
            "eval_success_rates": callback.eval_success_rates,
            "eval_mean_steps": callback.eval_mean_steps,
        }

    def predict(self, obs, deterministic=True, **kwargs):
        if self.model is None:
            raise ValueError("SACAgent has no model. Build, train, or load it first.")
        return self.model.predict(obs, deterministic=deterministic, **kwargs)

    def select_action(self, state, greedy=False, deterministic=None):
        deterministic = greedy if deterministic is None else deterministic
        action, _ = self.predict(state, deterministic=deterministic)
        return action

    def evaluate(
        self,
        env_factory=None,
        n_episodes=100,
        max_steps=S4_MAX_STEPS,
        deterministic=True,
        seed=7_000,
        non_null_threshold=1e-3,
    ):
        eval_factory = env_factory or self.env_factory
        if eval_factory is None:
            raise ValueError("SACAgent.evaluate requires an env_factory.")
        return evaluate_continuous_policy(
            self,
            eval_factory,
            n_episodes=n_episodes,
            max_steps=max_steps,
            deterministic=deterministic,
            seed=seed,
            non_null_threshold=non_null_threshold,
        )

    def get_policy_grid(self, n_bins=40):
        from ..visualization.plots import get_continuous_policy_grid

        if self.model is None:
            raise ValueError("SACAgent has no model. Build, train, or load it first.")
        return get_continuous_policy_grid(self, n_bins=n_bins)

    def save(self, path):
        if self.model is None:
            raise ValueError("SACAgent has no model to save.")
        self.model.save(path)

    @classmethod
    def load(cls, path, env_factory, seed=S4_DEFAULT_SEED, verbose=0, **kwargs):
        model = load_sb3_continuous_model(
            cls.algorithm,
            path,
            env_factory,
            seed=seed,
            verbose=verbose,
        )
        return cls(
            env_factory=env_factory,
            seed=seed,
            verbose=verbose,
            model=model,
            **kwargs,
        )
