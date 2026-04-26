import numpy as np


class TabularQLearningAgent:
    """Q-Learning agent with uniform state-space discretization."""

    def __init__(self, n_bins=40, n_actions=3,
                 alpha=0.15, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.pos_bins = np.linspace(-1.2, 0.6, n_bins + 1)
        self.vel_bins = np.linspace(-0.07, 0.07, n_bins + 1)

        self.Q = np.zeros((n_bins, n_bins, n_actions))
        self.N = np.zeros((n_bins, n_bins, n_actions))

    def discretize(self, state):
        pos, vel = state
        pi = np.clip(np.digitize(pos, self.pos_bins) - 1, 0, self.n_bins - 1)
        vi = np.clip(np.digitize(vel, self.vel_bins) - 1, 0, self.n_bins - 1)
        return pi, vi

    def select_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        pi, vi = self.discretize(state)
        return int(np.argmax(self.Q[pi, vi]))

    def update(self, state, action, reward, next_state, done):
        pi,  vi  = self.discretize(state)
        npi, nvi = self.discretize(next_state)
        self.N[pi, vi, action] += 1
        current_q = self.Q[pi, vi, action]
        target = reward if done else reward + self.gamma * np.max(self.Q[npi, nvi])
        self.Q[pi, vi, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def get_policy_grid(self):
        return np.argmax(self.Q, axis=2)

    def get_value_grid(self):
        return np.max(self.Q, axis=2)

    def get_visit_grid(self):
        return np.sum(self.N, axis=2)

    def save(self, path):
        np.savez(path, Q=self.Q, N=self.N, epsilon=[self.epsilon])

    @classmethod
    def load(cls, path, **kwargs):
        data = np.load(path)
        n_bins = data['Q'].shape[0]
        agent = cls(n_bins=n_bins, **kwargs)
        agent.Q = data['Q']
        agent.N = data['N']
        agent.epsilon = float(data['epsilon'][0])
        return agent


class SarsaAgent(TabularQLearningAgent):
    """On-policy SARSA — identical to Q-Learning except the TD target uses
    the action actually taken in the next state instead of the greedy one."""

    def update(self, state, action, reward, next_state, next_action, done):
        pi,vi = self.discretize(state)
        npi, nvi = self.discretize(next_state)
        self.N[pi, vi, action] += 1
        current_q = self.Q[pi, vi, action]
        target = reward if done else reward + self.gamma * self.Q[npi, nvi, next_action]
        self.Q[pi, vi, action] += self.alpha * (target - current_q)
