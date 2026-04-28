import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..training.loops import (
    train_tabular, evaluate_tabular,
    train_sarsa,
    train_dqn, evaluate_dqn,
)


class Testbed:
    """Manages a single experimental scenario: opens a TensorBoard writer,
    runs training, collects results, and cleans up."""

    def __init__(self, name: str, log_dir: str = "runs"):
        self.name = name
        self.log_dir = os.path.join(log_dir, name.replace("/", "_"))
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.results = {}
        print(f"[Testbed] '{name}' — logs -> {self.log_dir}")

    def run_tabular(self, env_factory, agent, n_episodes=10_000,
                    max_steps=200, eval_every=500, n_eval=20, verbose=True,
                    eval_env_factory=None):
        """Train + evaluate a TabularQLearningAgent and store results.
        eval_env_factory: if set, evaluation uses this env instead of env_factory
        (useful when training with reward shaping but evaluating on default reward)."""
        eval_factory = eval_env_factory or env_factory
        rewards, eval_means, eval_stds, eval_eps = train_tabular(
            env_factory, agent,
            n_episodes=n_episodes, max_steps=max_steps,
            eval_every=eval_every, n_eval=n_eval,
            verbose=verbose,
            tb_writer=self.writer, tb_tag=self.name,
        )
        final = evaluate_tabular(eval_factory, agent, n_episodes=100)
        self._log_final(final)
        self.results["tabular"] = {
            "rewards": rewards, "eval_means": eval_means,
            "eval_stds": eval_stds, "eval_episodes": eval_eps, "final": final,
        }
        return self.results["tabular"]

    def run_sarsa(self, env_factory, agent, n_episodes=10_000,
                  max_steps=200, eval_every=500, n_eval=20, verbose=True,
                  eval_env_factory=None):
        """Train + evaluate a SarsaAgent and store results."""
        eval_factory = eval_env_factory or env_factory
        rewards, eval_means, eval_stds, eval_eps = train_sarsa(
            env_factory, agent,
            n_episodes=n_episodes, max_steps=max_steps,
            eval_every=eval_every, n_eval=n_eval,
            verbose=verbose,
            tb_writer=self.writer, tb_tag=self.name,
        )
        final = evaluate_tabular(eval_factory, agent, n_episodes=100)
        self._log_final(final)
        self.results["sarsa"] = {
            "rewards": rewards, "eval_means": eval_means,
            "eval_stds": eval_stds, "eval_episodes": eval_eps, "final": final,
        }
        return self.results["sarsa"]

    def run_dqn(self, env_factory, agent, n_steps=200_000,
                max_steps_ep=200, eval_every=10_000, n_eval=20, verbose=True):
        """Train + evaluate a DQNAgent and store results."""
        rewards, eval_means, eval_stds, eval_steps = train_dqn(
            env_factory, agent,
            n_steps=n_steps, max_steps_ep=max_steps_ep,
            eval_every=eval_every, n_eval=n_eval,
            verbose=verbose,
            tb_writer=self.writer, tb_tag=self.name,
        )
        final = evaluate_dqn(env_factory, agent, n_episodes=100)
        self._log_final(final)
        self.results["dqn"] = {
            "rewards": rewards, "eval_means": eval_means,
            "eval_stds": eval_stds, "eval_steps": eval_steps, "final": final,
        }
        return self.results["dqn"]

    def run_continuous(self, env_factory, agent, total_timesteps,
                       eval_every=10_000, n_eval=20, final_n_eval=100,
                       verbose=True, eval_env_factory=None,
                       training_eval_env_factory=None,
                       deterministic_eval=True, model_save_path=None):
        """Train + evaluate a SACAgent/TD3Agent and store S4-style metrics."""
        if getattr(agent, "env_factory", None) is None:
            agent.env_factory = env_factory

        run_name = self.name.replace("/", "_")
        train_eval_factory = training_eval_env_factory or env_factory
        final_eval_factory = eval_env_factory or env_factory
        train_result = agent.train(
            run_name=run_name,
            total_timesteps=total_timesteps,
            eval_env_factory=train_eval_factory,
            eval_freq=eval_every,
            n_eval_episodes=n_eval,
            deterministic_eval=deterministic_eval,
            model_save_path=model_save_path,
        )
        final_eval = agent.evaluate(
            final_eval_factory,
            n_episodes=final_n_eval,
            deterministic=deterministic_eval,
        )
        final_rewards = [ep["reward"] for ep in final_eval["episodes"]]
        self._log_final_continuous(final_eval["summary"])

        result_key = getattr(agent, "algorithm", "continuous").lower()
        self.results[result_key] = {
            **train_result,
            "eval_means": train_result["eval_mean_rewards"],
            "eval_stds": train_result["eval_std_rewards"],
            "eval_steps": train_result["eval_timesteps"],
            "final": final_rewards,
            "final_eval": final_eval,
            "summary": final_eval["summary"],
        }
        if verbose:
            summary = final_eval["summary"]
            print(
                f"[Testbed] {getattr(agent, 'algorithm', 'continuous')} eval — "
                f"reward={summary['mean_reward']:.2f} +/- {summary['std_reward']:.2f} | "
                f"success={summary['success_rate']:.1%} | "
                f"steps={summary['mean_steps']:.1f}"
            )
        return self.results[result_key]

    def _log_final(self, final_rewards):
        mean = np.mean(final_rewards)
        std = np.std(final_rewards)
        success = np.mean(np.array(final_rewards) > -200)
        self.writer.add_scalar(f"{self.name}/final_mean", mean, 0)
        self.writer.add_scalar(f"{self.name}/final_std", std, 0)
        self.writer.add_scalar(f"{self.name}/success_rate", success, 0)
        print(f"[Testbed] Final eval — mean={mean:.2f} +/- {std:.2f} | "
              f"success={success:.1%}")

    def _log_final_continuous(self, summary):
        self.writer.add_scalar(f"{self.name}/final_mean", summary["mean_reward"], 0)
        self.writer.add_scalar(f"{self.name}/final_std", summary["std_reward"], 0)
        self.writer.add_scalar(f"{self.name}/success_rate", summary["success_rate"], 0)
        self.writer.add_scalar(f"{self.name}/mean_steps", summary["mean_steps"], 0)
        self.writer.add_scalar(f"{self.name}/mean_fuel", summary["mean_fuel"], 0)
        self.writer.add_scalar(
            f"{self.name}/mean_non_null_actions",
            summary["mean_non_null_actions"],
            0,
        )
        print(f"[Testbed] Final eval — mean={summary['mean_reward']:.2f} +/- "
              f"{summary['std_reward']:.2f} | success={summary['success_rate']:.1%}")

    def close(self):
        self.writer.close()
        print(f"[Testbed] '{self.name}' closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
