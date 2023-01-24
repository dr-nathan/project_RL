import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import nn
from torch.distributions import Categorical, MultivariateNormal
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

    def _update_policy(self, batch):
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

    def update_policy(self, batch):
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                self._update_policy(batch)
            return

        # TODO: check if this works
        # CONCLUSION: Nope
        # if torch.backends.mps.is_available():
        #     with torch.backends.mps.device(0):
        #         self._update_policy(batch)
        #     return

        self._update_policy(batch)

    def train(self, n_episodes):
        for _ in tqdm(range(n_episodes)):
            state = self.env.reset()
            terminated = False

            batch: list[tuple[float | torch.Tensor, ...]] = []

            while not terminated:
                state_tensor = torch.tensor(state).float().unsqueeze(0)
                mean, std = self.policy_network(state_tensor)
                action = torch.normal(mean, std)
                # clip the action to the action space
                # TODO: check if this is actually necessary
                action = torch.tanh(action)

                next_state, reward, terminated, *_ = self.env.step(action.item())
                state = next_state

                batch.append((mean, std, action, reward))

            self.update_policy(batch)

        return self.env.episode_data

    def get_greedy_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        mean, _ = self.policy_network(state)
        # TODO: check if tanh is actually necessary
        return torch.tanh(mean).item()

    def validate(self, price_data: dict[datetime, float]):
        state = self.env.reset(price_data=price_data)
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

# SOURCE: https://github.com/nikhilbarhate99/PPO-PyTorch
################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full(
            (action_dim,), action_std_init * action_std_init
        ).to(DEVICE)

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std
        ).to(DEVICE)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(DEVICE)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        action_std_init=0.6,
    ):
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print(
            "--------------------------------------------------------------------------------------------"
        )
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print(
                "setting actor output action_std to min_action_std : ", self.action_std
            )
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(DEVICE)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(DEVICE)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(DEVICE)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(DEVICE)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            print(f"{loss.mean().item():.3f}")
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
