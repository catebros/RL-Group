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


# Environment factory functions
def make_s1():
    return gym.make('MountainCar-v0')

def make_s2():
    return Monitor(gym.make('MountainCarContinuous-v1'))

def make_s3():
    return DiscreteFuelWrapper(gym.make('MountainCar-v0'))

def make_s4():
    return Monitor(ContinuousStepsWrapper(gym.make('MountainCarContinuous-v1')))
