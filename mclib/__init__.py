# mclib — Mountain Car RL library
# Agents
from .agents.tabular import TabularQLearningAgent, SarsaAgent
from .agents.dqn import QNetwork, DQNAgent
from .agents.sac import SACAgent
from .agents.td3 import TD3Agent

# Environments
from .envs.wrappers import (
    DiscreteFuelWrapper,
    DiscreteFuelWrapperV2,
    ContinuousStepsWrapper,
    ContinuousActionUseWrapper,
    ContinuousShapedRewardWrapper,
    ContinuousLinearActionWrapper,
    EnergyShapingWrapper,
    make_s1,
    make_s1_shaped,
    make_s2,
    make_s3,
    make_s3_v2,
    make_s4,
    make_s4_default,
    make_s4_action_use,
    make_s4_shaped,
    make_s4_linear_action,
)

# Training
from .training.loops import (
    train_tabular, evaluate_tabular,
    train_sarsa,
    train_dqn, evaluate_dqn,
)
from .training.continuous import (
    S4_DEFAULT_SEED,
    S4_TOTAL_TIMESTEPS,
    S4_TD3_TOTAL_TIMESTEPS,
    S4_MAX_STEPS,
    TD3_ACTION_NOISE_SIGMA,
    TD3_ACTION_NOISE_TYPE,
    TD3_GAMMA,
    TD3_LEARNING_STARTS,
    empty_continuous_eval_trace,
    load_continuous_eval_trace,
    make_continuous_eval_callback,
    make_sb3_continuous_model,
    load_sb3_continuous_model,
    train_sb3_continuous,
    evaluate_continuous_policy,
    summarize_continuous_metrics,
)


# Testbed
from .testbed.runner import Testbed

# Visualization
from .visualization.plots import (
    plot_training_curve,
    plot_policy_heatmap,
    plot_continuous_action_heatmap,
    plot_value_surface_3d,
    collect_trajectories,
    plot_phase_portrait,
    smooth,
    count_steps,
    get_sac_policy_grid,
    get_continuous_policy_grid,
    ACTION_CMAP,
    ACTION_COLORS,
    ACTION_LABELS,
)

# Analysis
from .analysis.explain import explain_policy_regression, multinomial_logreg
