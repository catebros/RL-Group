import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

ACTION_COLORS = {0: 'red', 1: 'cyan', 2: 'green'}
ACTION_LABELS = {0: 'Thrust Left', 1: 'No Thrust', 2: 'Thrust Right'}
ACTION_CMAP = mcolors.ListedColormap(['red', 'cyan', 'green'])


def smooth(data, window=50):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_training_curve(rewards, eval_x=None, eval_means=None, eval_stds=None,
                        title='Training Curve', window=100, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(rewards, alpha=0.2, color='steelblue', linewidth=0.5, label='Episode reward')
    if len(rewards) >= window:
        s = smooth(rewards, window)
        ax.plot(range(window - 1, len(rewards)), s, color='steelblue',
                linewidth=2, label=f'Smoothed (w={window})')

    if eval_x is not None and eval_means:
        ax.errorbar(eval_x, eval_means, yerr=eval_stds,
                    color='orange', fmt='o-', linewidth=2,
                    capsize=4, label='Eval mean +/- std')

    ax.set_title(title)
    ax.set_xlabel('Episode / Step')
    ax.set_ylabel('Total Reward')
    ax.legend(fontsize=9)
    return ax


def plot_policy_heatmap(policy_grid, n_bins=40, title='Policy Map', ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    pos_edges = np.linspace(-1.2,  0.6,  n_bins + 1)
    vel_edges = np.linspace(-0.07, 0.07, n_bins + 1)

    ax.pcolormesh(pos_edges, vel_edges, policy_grid.T,
                  cmap=ACTION_CMAP, vmin=0, vmax=2)

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(['red', 'cyan', 'green'],
                               ['Thrust Left', 'Idle', 'Thrust Right'])]
    ax.legend(handles=patches, loc='upper left', fontsize=8)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title(title)
    ax.axvline(0.5, color='gold', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':',linewidth=1)
    return ax


def plot_continuous_action_heatmap(action_grid, n_bins=40, title='Continuous Action Map',
                                   ax=None, magnitude=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    pos_edges = np.linspace(-1.2, 0.6, n_bins + 1)
    vel_edges = np.linspace(-0.07, 0.07, n_bins + 1)
    values = np.abs(action_grid) if magnitude else action_grid
    cmap = 'viridis' if magnitude else 'coolwarm'
    vmin, vmax = (0, 1) if magnitude else (-1, 1)

    mesh = ax.pcolormesh(pos_edges, vel_edges, values.T, cmap=cmap, vmin=vmin, vmax=vmax)
    label = '|action|' if magnitude else 'action'
    ax.figure.colorbar(mesh, ax=ax, label=label)
    ax.axvline(0.45, color='gold', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title(title)
    return ax


def plot_value_surface_3d(value_grid, n_bins=40, title='Value Function', ax=None):
    pos_vals = np.linspace(-1.2, 0.6, n_bins)
    vel_vals = np.linspace(-0.07, 0.07, n_bins)
    P, V = np.meshgrid(pos_vals, vel_vals)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(P, V, value_grid.T, cmap='viridis', alpha=0.85, edgecolor='none')
    ax.set_xlabel('Position x', labelpad=5)
    ax.set_ylabel('Velocity v', labelpad=5)
    ax.set_zlabel('Max Q-value')
    ax.set_title(title)
    return surf


def collect_trajectories(env_factory, agent_fn, n_episodes=10, max_steps=200):
    trajectories = []
    total_rewards = []
    env = env_factory()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=3000 + ep)
        traj= [obs.copy()]
        total = 0.0
        for _ in range(max_steps):
            action = agent_fn(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            traj.append(obs.copy())
            total += r
            if terminated or truncated:
                break
        trajectories.append(np.array(traj))
        total_rewards.append(total)
    env.close()
    return trajectories, total_rewards


def plot_phase_portrait(trajectories, rewards, title='Phase Portrait', ax=None,
                        goal_position=0.5):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    for traj, r, c in zip(trajectories, rewards, colors):
        ax.scatter(traj[:, 0], traj[:, 1], s=8, color=c, alpha=0.7, label=f'R={r:.1f}')
        ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.3, linewidth=0.8)

    ax.axvline(goal_position, color='gold', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':',linewidth=1)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper left')
    return ax


def count_steps(env_factory, agent_fn, n_episodes=100, max_steps=999):
    steps_list = []
    success_list = []
    env = env_factory()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=5000 + ep)
        for step in range(max_steps):
            action = agent_fn(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                steps_list.append(step + 1)
                success_list.append(True)
                break
            if truncated:
                steps_list.append(max_steps)
                success_list.append(False)
                break
        else:
            steps_list.append(max_steps)
            success_list.append(False)
    env.close()
    return steps_list, success_list


def get_sac_policy_grid(model, n_bins=40):
    return get_continuous_policy_grid(model, n_bins=n_bins)


def get_continuous_policy_grid(model, n_bins=40):
    pos_vals = np.linspace(-1.2, 0.6, n_bins)
    vel_vals = np.linspace(-0.07, 0.07, n_bins)
    actions  = np.zeros((n_bins, n_bins))
    for i, p in enumerate(pos_vals):
        for j, v in enumerate(vel_vals):
            obs = np.array([[p, v]], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            actions[i, j] = float(np.asarray(action).reshape(-1)[0])
    return actions


def plot_reward_distributions(agents_data, title='Final Reward Distributions', ncols=None):
    """
    agents_data: list of (label, rewards, color)
    """
    n = len(agents_data)
    if ncols is None:
        ncols = 2 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 4))
    axes = np.array(axes).flatten()

    for ax, (label, final, color) in zip(axes, agents_data):
        final = np.array(final)
        ax.hist(final, bins=20, color=color, alpha=0.7, edgecolor='black')
        mean = np.mean(final)
        ci = 1.96 * np.std(final) / np.sqrt(len(final))
        ax.axvline(mean, color='red', linestyle='--', label=f'mean={mean:.1f}')
        ax.axvline(mean - ci, color='gray', linestyle=':', label=f'95% CI +/-{ci:.1f}')
        ax.axvline(mean + ci, color='gray', linestyle=':')
        ax.set_title(f'{label}: Reward Distribution')
        ax.set_xlabel('Episode Reward')
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_convergence_and_boxplot(tabular_series, finals_data, title='Convergence & Final Performance'):
    """
    tabular_series: list of (results_dict, label, color, marker) for convergence lines.
                    results_dict must have 'eval_episodes', 'eval_means', 'eval_stds'.
    finals_data:    list of (label, rewards, color) for the boxplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    for results, label, color, marker in tabular_series:
        eps   = results['eval_episodes']
        means = np.array(results['eval_means'])
        stds  = np.array(results['eval_stds'])
        ax.plot(eps, means, f'{marker}-', color=color, label=label, linewidth=1.5)
        ax.fill_between(eps, means - stds, means + stds, alpha=0.15, color=color)
    ax.axhline(-200, color='black', linestyle=':', alpha=0.4, label='Timeout baseline')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Mean Reward (±1 std)')
    ax.set_title('Tabular Agents: Convergence Comparison')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    data    = [np.array(r) for _, r, _ in finals_data]
    labels  = [l for l, _, _ in finals_data]
    colors  = [c for _, _, c in finals_data]
    bp = ax2.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(-200, color='red', linestyle='--', alpha=0.4, label='Timeout')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Final Performance Distribution')
    ax2.legend(fontsize=9)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_visit_frequency(agents, labels, title='Exploration Coverage', n_bins=40, prefix=''):
    """
    agents: list of tabular agents with a get_visit_grid() method.
    """
    n = len(agents)
    fig, axes = plt.subplots(1, n, figsize=(n * 5 + 1, 4))
    axes = np.array(axes).flatten()

    pos_edges = np.linspace(-1.2, 0.6,  n_bins + 1)
    vel_edges = np.linspace(-0.07, 0.07, n_bins + 1)

    for ax, agent, label in zip(axes, agents, labels):
        visit = np.log1p(agent.get_visit_grid())
        im = ax.pcolormesh(pos_edges, vel_edges, visit.T, cmap='Blues')
        plt.colorbar(im, ax=ax, label='log(visits + 1)')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        pfx = f'{prefix} ' if prefix else ''
        ax.set_title(f'{pfx}{label}: State Visit Frequency')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig
