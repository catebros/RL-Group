# mclib — Mountain Car RL library
# Agents
from .agents.tabular import TabularQLearningAgent, SarsaAgent
from .agents.dqn import QNetwork, DQNAgent

# Environments
from .envs.wrappers import (
    DiscreteFuelWrapper, DiscreteFuelWrapperV2, ContinuousStepsWrapper, EnergyShapingWrapper,
    make_s1, make_s1_shaped, make_s2, make_s3, make_s3_v2, make_s4,
)

# Training
from .training.loops import (
    train_tabular, evaluate_tabular,
    train_sarsa,
    train_dqn, evaluate_dqn,
)


# Testbed
from .testbed.runner import Testbed

# Visualization
from .visualization.plots import (
    plot_training_curve,
    plot_policy_heatmap,
    plot_value_surface_3d,
    collect_trajectories,
    plot_phase_portrait,
    smooth,
    count_steps,
    get_sac_policy_grid,
    ACTION_CMAP,
    ACTION_COLORS,
    ACTION_LABELS,
)

# Analysis
from .analysis.explain import explain_policy_regression, multinomial_logreg
