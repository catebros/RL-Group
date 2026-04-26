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

    def _log_final(self, final_rewards):
        mean = np.mean(final_rewards)
        std = np.std(final_rewards)
        success = np.mean(np.array(final_rewards) > -200)
        self.writer.add_scalar(f"{self.name}/final_mean", mean, 0)
        self.writer.add_scalar(f"{self.name}/final_std", std, 0)
        self.writer.add_scalar(f"{self.name}/success_rate", success, 0)
        print(f"[Testbed] Final eval — mean={mean:.2f} +/- {std:.2f} | "
              f"success={success:.1%}")

    def close(self):
        self.writer.close()
        print(f"[Testbed] '{self.name}' closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
