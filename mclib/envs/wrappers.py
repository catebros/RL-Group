import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


class DiscreteFuelWrapper(gym.Wrapper):
    """
    Scenario 3: Discrete — Minimum Fuel (Variant A).
    Reward: -1 per step (like S1) + fuel bonus at goal.
    Fuel bonus = +100 - fuel_used (so less fuel = higher bonus).
    This ensures goal-reaching is priority while rewarding fuel efficiency.
    """
    BASE_GOAL_BONUS = 100.0
    STEP_COST = -1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.fuel_used = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Track fuel usage
        if action != 1:  # Thrust action
            self.fuel_used += 1
        
        # Step cost (same as S1)
        reward = self.STEP_COST
        
        # Goal bonus that depends on fuel used (less fuel = more bonus)
        if terminated:
            fuel_bonus = max(0, self.BASE_GOAL_BONUS - self.fuel_used)
            reward += fuel_bonus
        
        return obs, reward, terminated, truncated, info


class DiscreteFuelWrapperV2(gym.Wrapper):
    """
    Scenario 3: Discrete — Minimum Fuel (Variant B).
    Alternative reward design: fuel penalty per step + large fixed goal bonus.
    Reward: -0.5 if thrusting, 0 if idling, -0.1 per step time penalty, +100 on goal.
    This variant has lower fuel penalty but stronger time pressure.
    """
    GOAL_BONUS = 100.0
    FUEL_PENALTY = -0.5  # Lower than V1's implicit -1 per thrust
    TIME_PENALTY = -0.1  # Stronger time pressure

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Fuel penalty: -0.5 for thrust, 0 for idle
        fuel_cost = self.FUEL_PENALTY if action != 1 else 0.0
        
        # Time penalty per step
        time_cost = self.TIME_PENALTY
        
        # Fixed goal bonus (unlike V1's fuel-dependent bonus)
        bonus = self.GOAL_BONUS if terminated else 0.0
        
        reward = fuel_cost + time_cost + bonus
        
        return obs, reward, terminated, truncated, info


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


class ContinuousFuelShapedWrapper(gym.Wrapper):
    """
    Scenario 2: Continuous — Fuel-Minimizing with momentum shaping.
    Keeps the native squared-action penalty (-0.1*a^2 + 100 on goal) and adds
    small progress and velocity bonuses to guide exploration toward the
    momentum-building swing strategy without altering the fuel-cost objective.
    """
    PROGRESS_WEIGHT = 0.5
    VELOCITY_WEIGHT = 0.05

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=float)
        return obs, info

    def step(self, action):
        previous_obs = getattr(self, "_last_obs", None)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if previous_obs is not None:
            next_obs = np.asarray(obs, dtype=float)
            progress_bonus = self.PROGRESS_WEIGHT * float(next_obs[0] - previous_obs[0])
            velocity_bonus = self.VELOCITY_WEIGHT * abs(float(next_obs[1]))
            reward += progress_bonus + velocity_bonus
        self._last_obs = np.asarray(obs, dtype=float)
        return obs, reward, terminated, truncated, info


class EnergyShapingWrapper(gym.Wrapper):
    """
    Potential-based reward shaping for discrete MountainCar (Scenario 1 variant).
    Adds phi(s') - phi(s) where phi(s) = position + 0.5 * velocity ^2
    Guaranteed not to change the optimal policy (Ng et al. 1999).
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        prev_pos, prev_vel = self._prev_obs
        pos, vel = obs
        shaping = (pos + 0.5 * vel ** 2) - (prev_pos + 0.5 * prev_vel ** 2)
        self._prev_obs = obs
        return obs, reward + shaping, terminated, truncated, info


# Environment factory functions
def make_s1():
    return gym.make('MountainCar-v0')


def make_s1_shaped():
    return EnergyShapingWrapper(gym.make('MountainCar-v0'))


def make_s2():
    return Monitor(gym.make('MountainCarContinuous-v0'))


def make_s2_shaped():
    return Monitor(ContinuousFuelShapedWrapper(gym.make('MountainCarContinuous-v0')))


def make_s3():
    return DiscreteFuelWrapper(gym.make('MountainCar-v0'))


def make_s3_v2():
    """Alternative fuel reward variant with lower fuel penalty but time pressure."""
    return DiscreteFuelWrapperV2(gym.make('MountainCar-v0'))


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
