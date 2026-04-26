import numpy as np


def train_tabular(env_factory, agent, n_episodes=10_000, max_steps=200,
                  eval_every=500, n_eval=20, verbose=True,
                  tb_writer=None, tb_tag="tabular"):
    """Train a TabularQLearningAgent."""
    rewards_hist = []
    eval_means = []
    eval_stds = []
    eval_episodes = []

    env = env_factory()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        rewards_hist.append(total_reward)

        if tb_writer is not None:
            tb_writer.add_scalar(f"{tb_tag}/episode_reward", total_reward, ep)
            tb_writer.add_scalar(f"{tb_tag}/epsilon", agent.epsilon, ep)

        if (ep + 1) % eval_every == 0:
            eval_r = evaluate_tabular(env_factory, agent, n_episodes=n_eval)
            eval_means.append(np.mean(eval_r))
            eval_stds.append(np.std(eval_r))
            eval_episodes.append(ep + 1)

            if tb_writer is not None:
                tb_writer.add_scalar(f"{tb_tag}/eval_mean", eval_means[-1], ep + 1)
                tb_writer.add_scalar(f"{tb_tag}/eval_std", eval_stds[-1], ep + 1)

            if verbose:
                print(f" Ep {ep+1:6d} | train_r={total_reward:7.1f} "
                      f"| eval_mean={eval_means[-1]:7.2f} +/- {eval_stds[-1]:.2f} "
                      f"| eps={agent.epsilon:.3f}")

    env.close()
    return rewards_hist, eval_means, eval_stds, eval_episodes


def evaluate_tabular(env_factory, agent, n_episodes=100, max_steps=200):
    """Evaluate a tabular agent greedily; returns list of episode rewards."""
    rewards = []
    env = env_factory()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total = 0.0
        for _ in range(max_steps):
            action = agent.select_action(obs, greedy=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards


def train_sarsa(env_factory, agent, n_episodes=10_000, max_steps=200,
                eval_every=500, n_eval=20, verbose=True,
                tb_writer=None, tb_tag="sarsa"):
    """Train a SarsaAgent (on-policy TD control)."""
    rewards_hist= []
    eval_means = []
    eval_stds  = []
    eval_episodes = []

    env = env_factory()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        action = agent.select_action(obs)
        total_reward = 0.0

        for _ in range(max_steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_obs)
            agent.update(obs, action, reward, next_obs, next_action, done)
            obs = next_obs
            action = next_action
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        rewards_hist.append(total_reward)

        if tb_writer is not None:
            tb_writer.add_scalar(f"{tb_tag}/episode_reward", total_reward, ep)
            tb_writer.add_scalar(f"{tb_tag}/epsilon", agent.epsilon, ep)

        if (ep + 1) % eval_every == 0:
            eval_r = evaluate_tabular(env_factory, agent, n_episodes=n_eval)
            eval_means.append(np.mean(eval_r))
            eval_stds.append(np.std(eval_r))
            eval_episodes.append(ep + 1)

            if tb_writer is not None:
                tb_writer.add_scalar(f"{tb_tag}/eval_mean", eval_means[-1], ep + 1)
                tb_writer.add_scalar(f"{tb_tag}/eval_std",eval_stds[-1],ep + 1)

            if verbose:
                print(f"Ep {ep+1:6d} | train_r={total_reward:7.1f} "
                      f"| eval_mean={eval_means[-1]:7.2f} +/- {eval_stds[-1]:.2f} "
                      f"| eps={agent.epsilon:.3f}")

    env.close()
    return rewards_hist, eval_means, eval_stds, eval_episodes


def train_dqn(env_factory, agent, n_steps=200_000, max_steps_ep=200,
              eval_every=10_000, n_eval=20, verbose=True,
              tb_writer=None, tb_tag="dqn"):
    """Train a DQNAgent for a fixed number of environment steps."""
    rewards_hist = []
    eval_means = []
    eval_stds = []
    eval_steps = []
    total_steps = 0
    episode = 0

    env = env_factory()

    while total_steps < n_steps:
        obs, _ = env.reset(seed=episode)
        ep_reward = 0.0
        episode += 1

        for _ in range(max_steps_ep):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            obs = next_obs
            ep_reward += reward
            total_steps += 1

            if tb_writer is not None and loss is not None:
                tb_writer.add_scalar(f"{tb_tag}/loss", loss, total_steps)
                tb_writer.add_scalar(f"{tb_tag}/epsilon", agent.epsilon, total_steps)

            if done:
                break

            if total_steps % eval_every == 0:
                eval_r = evaluate_dqn(env_factory, agent, n_episodes=n_eval)
                eval_means.append(np.mean(eval_r))
                eval_stds.append(np.std(eval_r))
                eval_steps.append(total_steps)

                if tb_writer is not None:
                    tb_writer.add_scalar(f"{tb_tag}/eval_mean",eval_means[-1], total_steps)
                    tb_writer.add_scalar(f"{tb_tag}/eval_std", eval_stds[-1], total_steps)

                if verbose:
                    print(f" Step {total_steps:7d} | ep={episode:5d} "
                          f"| eval_mean={eval_means[-1]:7.2f} +/- {eval_stds[-1]:.2f} "
                          f"| eps={agent.epsilon:.3f}")

        rewards_hist.append(ep_reward)

        if tb_writer is not None:
            tb_writer.add_scalar(f"{tb_tag}/episode_reward", ep_reward, episode)

    env.close()
    return rewards_hist, eval_means, eval_stds, eval_steps


def evaluate_dqn(env_factory, agent, n_episodes=100, max_steps=200):
    """Evaluate a DQN agent greedily; returns list of episode rewards."""
    rewards = []
    env = env_factory()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        total = 0.0
        for _ in range(max_steps):
            action = agent.select_action(obs, greedy=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards
