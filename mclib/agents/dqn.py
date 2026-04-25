import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Fully-connected Q-value network."""

    def __init__(self, state_dim=2, action_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(self, state_dim=2, action_dim=3,
                 lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay=5000,
                 buffer_size=50_000, batch_size=64,
                 target_update_freq=500):
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_freq = target_update_freq
        self.steps_done = 0

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps_done / self.eps_decay)

    def select_action(self, state, greedy=False):
        self.steps_done += 1
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(state_t)
        return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        curr_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def get_q_values_grid(self, n_bins=40):
        pos_vals = np.linspace(-1.2, 0.6, n_bins)
        vel_vals = np.linspace(-0.07, 0.07, n_bins)
        policy = np.zeros((n_bins, n_bins), dtype=int)
        values = np.zeros((n_bins, n_bins))
        for i, p in enumerate(pos_vals):
            for j, v in enumerate(vel_vals):
                s = torch.FloatTensor([[p, v]])
                with torch.no_grad():
                    q = self.policy_net(s)[0].numpy()
                policy[i, j] = int(np.argmax(q))
                values[i, j] = float(np.max(q))
        return policy, values
