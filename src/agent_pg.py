import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.environment import ContinuousDamEnv
from src.utils import discounted_reward
import numpy as np

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

torch.autograd.set_detect_anomaly(True)


class BasicPGNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_std = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        std = torch.exp(std)
        std = torch.clamp(std, min=1e-4)  # std cannot be < 0
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

    def update_policy(self, batch, sample_size: int = 1000):
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
        self,
        env: ContinuousDamEnv,
        learning_rate: float = 2.5e-4,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 512,
        discount_factor: float = 0.98,
        entropy_loss_coeff: float = 0.01,
    ):
        self.env = env
        self.state_size, *_ = env.observation_space.shape
        self.action_size, *_ = env.action_space.shape

        self.policy_network = BasicPGNetwork(self.state_size, self.action_size).to(
            DEVICE
        )

        # training parameters
        # TODO: check if AdamW is required
        self.optimizer = torch.optim.Adagrad(
            self.policy_network.parameters(), lr=learning_rate
        )
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.entropy_loss_coeff = entropy_loss_coeff

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0)
            mean, std = self.policy_network(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_probs = dist.log_prob(action)

        return action.item(), log_probs

    def calculate_loss(self, states, actions, rewards, old_log_probs):
        # action log probs ratio
        means, stds = self.policy_network(states)
        curr_dist = torch.distributions.Normal(means, stds)
        curr_log_probs = curr_dist.log_prob(actions)

        log_prob_ratios = torch.exp(curr_log_probs - old_log_probs)

        curr_entropy = curr_dist.entropy()

        # advantage TODO: check if this is correct
        advantages = rewards - rewards.mean()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # surrogates
        surr1 = advantages * log_prob_ratios
        surr2 = advantages * torch.clamp(
            log_prob_ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )
        surr_loss = torch.min(surr1, surr2)

        # loss
        loss = (-surr_loss - self.entropy_loss_coeff * curr_entropy).mean()

        return loss

    def update_policy(
        self,
        states,
        actions,
        rewards,
        old_log_probs,
    ):
        for _ in range(self.epochs):
            indexes = torch.randperm(len(self.env))

            for start in range(0, len(self.env), self.batch_size):
                batch_indexes = indexes[start : start + self.batch_size]

                batch_states = states[batch_indexes]
                batch_actions = actions[batch_indexes]
                batch_rewards = torch.tensor(
                    discounted_reward(
                        rewards[batch_indexes].detach().cpu(), self.discount_factor
                    ),
                    device=DEVICE,
                )
                batch_log_probs = old_log_probs[batch_indexes]

                self.policy_network = self.policy_network.to(DEVICE)

                loss = self.calculate_loss(
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_log_probs,
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                # apply_grad_clipping(self.policy_network, 0.5)

                self.optimizer.step()

                self._pbar.set_description(f"Loss: {loss.item():.4f}")

    def train(
        self,
        n_episodes: int,
        save_path: None | Path = None,
        save_frequency: int = 10,
    ):
        self.policy_network.train()
        self._pbar = tqdm(range(n_episodes))
        for i in self._pbar:

            state, _ = self.env.reset()
            terminated = False
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            self.policy_network = self.policy_network.cpu()

            while not terminated:
                action, log_probs = self.get_action(state)
                next_state, reward, terminated, *_ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_log_probs.append(log_probs)
                state = next_state

            states_tensor = torch.tensor(states, device=DEVICE).float()
            actions_tensor = torch.tensor(actions, device=DEVICE).float().unsqueeze(1)
            rewards_tensor = torch.tensor(rewards, device=DEVICE).float()
            old_log_probs_tensor = torch.tensor(old_log_probs, device=DEVICE).float()

            self.update_policy(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                old_log_probs_tensor,
            )

            reward = self.env.episode_data.total_reward

            if save_path is not None and i % save_frequency == 0:
                self.save(save_path / f"{reward:.0f}.pt")

    def validate(self, price_data: dict[datetime, float]):
        state, _ = self.env.reset(price_data=price_data)
        terminated = False

        while not terminated:
            action, *_ = self.get_action(state)
            next_state, _, terminated, *_ = self.env.step(action)
            state = next_state

        return self.env.episode_data

    def save(self, path: str | Path):
        torch.save(self.policy_network.state_dict(), path)

    def load(self, path: str | Path):
        self.policy_network.load_state_dict(torch.load(path))
