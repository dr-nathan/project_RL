import math
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from src.environment import ContinuousDamEnv

# DEVICE = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

# it's generally actually not beneficial to use a GPU for this, as steps are not taken in parallel
# so single-core performance is more important. during backpropagation a GPU can be faster, but the
# overhead of copying data to and from the GPU is not worth it
DEVICE = torch.device("cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_std = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.fc_std(x))
        return mean, std


class PolicyGradientAgent:
    def __init__(self, learning_rate: float, env: ContinuousDamEnv):
        self.env = env
        self.state_size, *_ = env.observation_space.shape
        self.action_size, *_ = env.action_space.shape

        self.policy_network = PolicyNetwork(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=learning_rate
        )

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        mean, std = self.policy_network(state)
        action = torch.normal(mean, std)
        return action.item()

    def update_policy(self, batch):
        # TODO: make cuda optional
        with torch.cuda.device(0):
            # compute the probabilities of the actions taken under the current policy
            means = torch.stack([e[0] for e in batch])
            stds = torch.stack([e[1] for e in batch])
            actions = torch.stack([e[2] for e in batch])
            rewards = torch.vstack([torch.tensor(e[3]) for e in batch])

            log_probs = self._get_log_prob(means, stds, actions)

            returns = log_probs * rewards.sum()
            policy_loss = -returns.mean()
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

    def train(self, n_episodes):
        for _ in tqdm(range(n_episodes)):
            state = self.env.reset()
            terminated = False

            batch: list[tuple[float | torch.Tensor, ...]] = []

            while not terminated:
                state_tensor = torch.tensor(state).float().unsqueeze(0)
                mean, std = self.policy_network(state_tensor)
                action = torch.normal(mean, std)

                next_state, reward, terminated, *_ = self.env.step(action.item())
                state = next_state

                batch.append((mean, std, action, reward))

            self.update_policy(batch)

        return self.env.episode_data

    def validate(self, price_data: dict[datetime, float]):
        state = self.env.reset(price_data=price_data)
        terminated = False

        while not terminated:
            action = self.get_action(state)
            next_state, _, terminated, *_ = self.env.step(action)
            state = next_state

        return self.env.episode_data

    def _get_log_prob(self, mean, std, action):
        log_std = torch.log(std)
        var = std.pow(2)
        log_prob = (
            -0.5 * ((action - mean) / var).pow(2)
            - 0.5 * math.log(2 * math.pi)
            - log_std
        )
        return log_prob.sum(dim=-1)

    def save(self, path: str | Path):
        torch.save(self.policy_network.state_dict(), path)

    def load(self, path: str | Path):
        self.policy_network.load_state_dict(torch.load(path))
