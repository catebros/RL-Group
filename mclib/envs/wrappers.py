import gymnasium as gym
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

def make_s3_v2():
    """Alternative fuel reward variant with lower fuel penalty but time pressure."""
    return DiscreteFuelWrapperV2(gym.make('MountainCar-v0'))

def make_s4():
    return Monitor(ContinuousStepsWrapper(gym.make('MountainCarContinuous-v1')))
