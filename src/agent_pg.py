import math
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.environment import ContinuousDamEnv

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# it's generally actually not beneficial to use a GPU for this, as steps are not taken in parallel
# so single-core performance is more important. during backpropagation a GPU can be faster, but the
# overhead of copying data to and from the GPU is not worth it
# DEVICE = torch.device("cpu")

print(f"{DEVICE = }")


class BasicPGNetwork(nn.Module):
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
        std = self.fc_std(x)
        std = torch.exp(std)
        std = torch.clamp(std, min=0)  # std cannot be < 0
        return mean, std


class BasicPGAgent:
    def __init__(self, learning_rate: float, env: ContinuousDamEnv):
        self.env = env
        self.state_size, *_ = env.observation_space.shape
        self.action_size, *_ = env.action_space.shape

        self.policy_network = BasicPGNetwork(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adamax(
            self.policy_network.parameters(), lr=learning_rate
        )

    def _update_policy(self, batch, sample_size: int = 1000):
        # get total reward
        rewards = torch.vstack([torch.tensor(e[3]) for e in batch])
        returns = rewards.sum()

        # sample actions from batch
        batch = random.sample(batch, sample_size)

        # compute the probabilities of the actions taken under the current policy
        means = torch.stack([e[0] for e in batch])
        stds = torch.stack([e[1] for e in batch])
        actions = torch.stack([e[2] for e in batch])

        log_probs = torch.distributions.Normal(means, stds).log_prob(actions)

        policy_loss = -(log_probs * returns).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()

    def update_policy(self, batch):
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                return self._update_policy(batch)

        # TODO: check if this works
        # CONCLUSION: Nope
        # if torch.backends.mps.is_available():
        #     with torch.backends.mps.device(0):
        #         return self._update_policy(batch)

        return self._update_policy(batch)

    def train(
        self, n_episodes: int, save_path: None | Path = None, save_frequency: int = 10
    ):
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            state, _ = self.env.reset()
            terminated = False

            batch: list[tuple[float | torch.Tensor, ...]] = []

            while not terminated:
                state_tensor = torch.tensor(state).float().unsqueeze(0)
                mean, std = self.policy_network(state_tensor)
                action = torch.normal(mean, std)

                next_state, reward, terminated, *_ = self.env.step(action.item())
                state = next_state

                batch.append((mean, std, action, reward))

            loss = self.update_policy(batch)
            reward = self.env.episode_data.total_reward
            pbar.set_description(f"Loss: {loss:.4f}, Reward: {reward:.4f}")

            if save_path is not None and i % save_frequency == 0:
                self.save(save_path / f"{reward:.0f}.pt")

        return self.env.episode_data

    def get_greedy_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        mean, _ = self.policy_network(state)
        return mean.item()

    def validate(self, price_data: dict[datetime, float]):
        state, _ = self.env.reset(price_data=price_data)
        terminated = False

        while not terminated:
            action = self.get_greedy_action(state)
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


class PPOAgent:
    def __init__(
        self, learning_rate: float, env: ContinuousDamEnv, clip_epsilon: float
    ):
        self.env = env
        self.state_size, *_ = env.observation_space.shape
        self.action_size, *_ = env.action_space.shape

        self.policy_network = BasicPGNetwork(self.state_size, self.action_size).to(
            DEVICE
        )
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=learning_rate
        )
        self.clip_epsilon = clip_epsilon
        self.old_policy_network = deepcopy(self.policy_network)

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        mean, std = self.policy_network(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action.item(), log_probs, dist.entropy()

    def _update_policy(self, states, actions, rewards, old_log_probs, entropy):
        for _ in range(5):
            mean, std = self.policy_network(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * rewards
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * rewards
            )
            policy_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy.mean()
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()
            self.old_policy_network.load_state_dict(self.policy_network.state_dict())

    def update_policy(self, *args, **kwargs):
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                return self._update_policy(*args, **kwargs)

        return self._update_policy(*args, **kwargs)

    def train(self, n_episodes):
        for _ in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            terminated = False
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            entropies = []

            while not terminated:
                action, log_probs, entropy = self.get_action(state)
                next_state, reward, terminated, *_ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_log_probs.append(log_probs)
                entropies.append(entropy)
                state = next_state

            states = torch.tensor(states).float()
            actions = torch.tensor(actions).float().unsqueeze(1)
            rewards = torch.tensor(rewards).float()
            old_log_probs = torch.tensor(old_log_probs).float()
            entropies = torch.tensor(entropies).float()
            self.update_policy(states, actions, rewards, old_log_probs, entropies)
