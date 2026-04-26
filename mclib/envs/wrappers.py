import gymnasium as gym
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
    return Monitor(gym.make('MountainCarContinuous-v1'))

def make_s3():
    return DiscreteFuelWrapper(gym.make('MountainCar-v0'))

def make_s4():
    return Monitor(ContinuousStepsWrapper(gym.make('MountainCarContinuous-v1')))
