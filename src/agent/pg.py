from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.environment.dam import ContinuousDamEnv
from src.utils import discounted_reward

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print(f"Using {DEVICE = }")
torch.autograd.set_detect_anomaly(True)


class BasicPGNetwork(nn.Module):
    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 5, n_linear: int = 3
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fcs = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_linear)]
        )

        self.activation = nn.ReLU()

        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_std = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for fc in self.fcs:
            x = self.activation(fc(x))

        mean = self.fc_mean(x)
        std = self.fc_std(x)
        std = torch.exp(std)
        std = torch.clamp(std, min=1e-8)  # std cannot be < 0
        return mean, std


class PGAgentBase:
    def __init__(self, env: ContinuousDamEnv, hidden_size: int, hidden_layers: int):
        self.env = env
        self.state_size, *_ = env.observation_space.shape
        self.action_size, *_ = env.action_space.shape

        self.policy_network = BasicPGNetwork(
            self.state_size, self.action_size, hidden_size, hidden_layers
        ).to(DEVICE)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0)
            mean, std = self.policy_network(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_probs = dist.log_prob(action)

        return action.item(), log_probs

    def validate(self, price_data: dict[datetime, float]):
        self.policy_network = self.policy_network.cpu()
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
        self.policy_network.load_state_dict(torch.load(path, map_location=DEVICE))


class BasicPGAgent(PGAgentBase):
    def __init__(
        self,
        env: ContinuousDamEnv,
        discount_factor: float = 0.98,
        epochs: int = 5,
        hidden_layers: int = 3,
        hidden_size: int = 5,
        learning_rate: float = 1e-3,
    ):
        super().__init__(env, hidden_size, hidden_layers)

        self.optimizer = Adam(
            self.policy_network.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer)

        self.epochs = epochs
        self.gamma = discount_factor

    def calculate_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
    ):
        # compute the probabilities of the actions taken under the current policy
        means, stds = self.policy_network(states)
        log_probs = torch.distributions.Normal(means, stds).log_prob(actions)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # compute the loss for a policy agent
        loss = -torch.mean(log_probs * advantages)

        return loss

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
    ):
        self.policy_network = self.policy_network.to(DEVICE)
        epoch_losses = []

        for _ in range(self.epochs):
            loss = self.calculate_loss(states, actions, rewards, advantages)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())

        rewards_sum = rewards.sum().item()
        loss_mean = sum(epoch_losses) / len(epoch_losses)
        self._pbar.set_description(f"Loss: {loss_mean:.4f}, Rewards: {rewards_sum}")

        self.scheduler.step(-rewards_sum)

    def play_game(self):
        with torch.no_grad():
            self.policy_network = self.policy_network.cpu()
            state, _ = self.env.reset()
            terminated = False

            states, actions, rewards = [], [], []

            while not terminated:
                action, *_ = self.get_action(state)
                next_state, reward, terminated, *_ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            advantages = discounted_reward(rewards, self.gamma)

        return states, actions, rewards, advantages

    def train(
        self, n_episodes: int, save_path: None | Path = None, save_frequency: int = 5
    ):
        self._pbar = tqdm(range(n_episodes))
        for i in self._pbar:
            states, actions, rewards, advantages = self.play_game()
            self.update_policy(
                torch.tensor(states, device=DEVICE).float(),
                torch.tensor(actions, device=DEVICE).float().unsqueeze(1),
                torch.tensor(rewards, device=DEVICE).float(),
                torch.tensor(advantages, device=DEVICE).float(),
            )

            if save_path is not None and i % save_frequency == 0:
                reward = self.env.episode_data.total_reward
                self.save(save_path / f"{i}-{reward=:.0f}.pt")


class PPOAgent(PGAgentBase):
    def __init__(
        self,
        env: ContinuousDamEnv,
        batch_size: int = 1024,
        clip_epsilon: float = 0.2,
        discount_factor: float = 0.98,
        entropy_loss_coeff: float = 0.01,
        epochs: int = 5,
        hidden_layers: int = 3,
        hidden_size: int = 5,
        learning_rate: float = 1e-3,
    ):
        super().__init__(env, hidden_size, hidden_layers)

        self.optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer)

        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon
        self.entropy_loss_coeff = entropy_loss_coeff
        self.epochs = epochs
        self.gamma = discount_factor

    def calculate_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ):
        # action log probs ratio
        means, stds = self.policy_network(states)
        curr_dist = torch.distributions.Normal(means, stds)
        curr_log_probs = curr_dist.log_prob(actions)

        log_prob_ratios = torch.exp(curr_log_probs - old_log_probs.detach())

        curr_entropy = curr_dist.entropy()

        # normalized advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # surrogates
        surr1 = advantages * log_prob_ratios
        surr2 = advantages * torch.clamp(
            log_prob_ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )
        surr_loss = -torch.min(surr1, surr2)

        # loss
        loss = torch.mean(surr_loss - self.entropy_loss_coeff * curr_entropy)

        return loss

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ):
        self.policy_network = self.policy_network.to(DEVICE)
        epoch_losses = []

        for _ in range(self.epochs):
            indexes = torch.randperm(len(states))
            epoch_losses = []

            for start in range(0, len(states), self.batch_size):
                batch_indexes = indexes[start : start + self.batch_size]

                batch_states = states[batch_indexes]
                batch_actions = actions[batch_indexes]
                batch_rewards = rewards[batch_indexes]
                batch_advantages = advantages[batch_indexes]
                batch_log_probs = old_log_probs[batch_indexes]

                self.policy_network = self.policy_network.to(DEVICE)

                loss = self.calculate_loss(
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_advantages,
                    batch_log_probs,
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                self.optimizer.step()

                epoch_losses.append(loss.item())

        rewards_sum = rewards.sum().item()
        loss_mean = sum(epoch_losses) / len(epoch_losses)
        self._pbar.set_description(f"Loss: {loss_mean:.4f}, Rewards: {rewards_sum}")

        self.scheduler.step(-rewards_sum)

    def play_game(self):
        with torch.no_grad():
            state, _ = self.env.reset()
            terminated = False
            states, actions, rewards, old_log_probs = [], [], [], []
            self.policy_network = self.policy_network.cpu()

            while not terminated:
                action, log_probs = self.get_action(state)
                next_state, reward, terminated, *_ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_log_probs.append(log_probs)
                state = next_state

            advantages = discounted_reward(rewards, self.gamma)

        return states, actions, rewards, advantages, old_log_probs

    def train(
        self,
        n_episodes: int,
        save_path: None | Path = None,
        save_frequency: int = 10,
    ):
        self.policy_network.train()
        self._pbar = tqdm(range(n_episodes))
        for i in self._pbar:
            (
                states,
                actions,
                rewards,
                advantages,
                old_log_probs,
            ) = self.play_game()

            self.update_policy(
                torch.tensor(states, device=DEVICE).float(),
                torch.tensor(actions, device=DEVICE).float().unsqueeze(1),
                torch.tensor(rewards, device=DEVICE).float(),
                torch.tensor(advantages, device=DEVICE).float(),
                torch.tensor(old_log_probs, device=DEVICE).float(),
            )

            if save_path is not None and i % save_frequency == 0:
                reward = self.env.episode_data.total_reward
                self.save(save_path / f"{i}-{reward=:.0f}.pt")
