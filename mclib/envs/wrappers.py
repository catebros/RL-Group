import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


class DiscreteFuelWrapper(gym.Wrapper):
    """
    Scenario 3: Discrete — Minimum Fuel.
    Reward: -1 for thrust actions (0=left, 2=right), 0 for coasting (1=idle).
    Bonus:+100 on goal termination.
    """
    GOAL_BONUS = 100.0

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        fuel_cost = -1.0 if action != 1 else 0.0
        bonus = self.GOAL_BONUS if terminated else 0.0
        return obs, fuel_cost + bonus, terminated, truncated, info


class ContinuousStepsWrapper(gym.Wrapper):
    """
    Scenario 4: Continuous — Minimum Steps.
    Reward: -1 per timestep + 100 on goal termination.
    """
    STEP_COST = -1.0
    GOAL_BONUS = 100.0

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.STEP_COST + (self.GOAL_BONUS if terminated else 0.0)
        return obs, reward, terminated, truncated, info


class ContinuousActionUseWrapper(gym.Wrapper):
    """
    Scenario 4: Continuous — Sparse action-use cost.
    Reward: small time penalty, non-null engine-use cost, small action magnitude
    cost, and +100 on goal termination.
    """
    ACTION_EPSILON = 1e-3
    TIME_PENALTY = 0.01
    ACTION_USE_COST = 0.10
    ACTION_MAG_COST = 0.02
    GOAL_BONUS = 100.0

    def step(self, action):
        applied_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        obs, _, terminated, truncated, info = self.env.step(action)
        action_abs = abs(applied_action)
        non_null = float(action_abs > self.ACTION_EPSILON)
        reward = (
            -self.TIME_PENALTY
            - self.ACTION_USE_COST * non_null
            - self.ACTION_MAG_COST * action_abs
            + (self.GOAL_BONUS if terminated else 0.0)
        )
        info = dict(info)
        info["s4_action_abs"] = action_abs
        info["s4_non_null_action"] = bool(non_null)
        return obs, reward, terminated, truncated, info


class ContinuousShapedRewardWrapper(ContinuousActionUseWrapper):
    """
    Scenario 4: Continuous — shaped action-use training reward.
    Adds progress and momentum feedback to the scenario action-use reward.
    """
    PROGRESS_WEIGHT = 1.0
    VELOCITY_WEIGHT = 0.1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=float)
        return obs, info

    def step(self, action):
        previous_obs = getattr(self, "_last_obs", None)
        obs, reward, terminated, truncated, info = super().step(action)

        if previous_obs is not None:
            next_obs = np.asarray(obs, dtype=float)
            progress_bonus = self.PROGRESS_WEIGHT * float(next_obs[0] - previous_obs[0])
            velocity_bonus = self.VELOCITY_WEIGHT * abs(float(next_obs[1]))
            reward += progress_bonus + velocity_bonus
            info = dict(info)
            info["s4_progress_bonus"] = progress_bonus
            info["s4_velocity_bonus"] = velocity_bonus

        self._last_obs = np.asarray(obs, dtype=float)
        return obs, reward, terminated, truncated, info


class ContinuousLinearActionWrapper(ContinuousActionUseWrapper):
    """Deprecated compatibility alias; use ContinuousActionUseWrapper."""


# Environment factory functions
def make_s1():
    return gym.make('MountainCar-v0')


def make_s2():
    return Monitor(gym.make('MountainCarContinuous-v0'))


def make_s3():
    return DiscreteFuelWrapper(gym.make('MountainCar-v0'))


def make_s4():
    return Monitor(ContinuousStepsWrapper(gym.make('MountainCarContinuous-v0')))


def make_s4_default():
    return Monitor(gym.make('MountainCarContinuous-v0'))


def make_s4_action_use():
    return Monitor(ContinuousActionUseWrapper(gym.make('MountainCarContinuous-v0')))


def make_s4_shaped():
    return Monitor(ContinuousShapedRewardWrapper(gym.make('MountainCarContinuous-v0')))


def make_s4_linear_action():
    return make_s4_action_use()
