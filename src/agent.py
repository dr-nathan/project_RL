from copy import deepcopy
import random
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# So you don't have to install torch if you're not using the PG agent
try:
    import torch
    from torch import nn

    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                            else "cpu")
except ImportError:
    torch = None


# create agent
class QLearnAgent:
    # TODO: look into making discount factor dynamic
    def __init__(self, env: gym.Env, discount_factor: float = 0.98):
        self.env = env
        self.discount_factor = discount_factor

        # create Q table
        self.Qtable = np.zeros(
            np.append(self.env.observation_space.nvec, self.env.action_space.n)
        )

    def update_Q_table(self, state, action, reward, next_state):
        q_next = self.Qtable[next_state].max()
        q_current_idx = *state, action

        # update Q table
        self.Qtable[q_current_idx] = (1 - self.alpha) * self.Qtable[
            q_current_idx
        ] + self.alpha * (reward + self.discount_factor * q_next)

    def make_decision(self, state, policy: str = "epsilon_greedy"):
        if policy == "greedy":
            return np.argmax(self.Qtable[state])

        if policy == "epsilon_greedy":
            return self.choose_action_eps_greedy(state)

        raise ValueError("Unknown policy")

    def choose_action_eps_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()

        # nonzero to make sure we don't default to action 0
        best_actions = np.flatnonzero(self.Qtable[state] == self.Qtable[state].max())
        if len(best_actions) == 1:
            return best_actions[0]

        return random.choice(best_actions)

    def train(
        self,
        policy,
        n_episodes: int,
        epsilon: float = 0.1,
        epsilon_decay: bool = False,
        alpha: float = 0.1,
        random_startpoint: bool = False,
        start_amount: float = 0.5,
        val_price_data:dict[datetime, float] | None = None,
    ):

        # intitialize stuff
        self.epsilon_decay = epsilon_decay
        self.train_reward = []
        self.val_reward = []

        if self.epsilon_decay:
            epsilon_start = 1
            epsilon_end = 0.01
            epsilon_decay_step = np.exp(
                np.log(epsilon_end / epsilon_start) / n_episodes
            )
        else:
            self.epsilon = epsilon

        self.alpha = alpha

        val_env = None
        if val_price_data is not None:
            val_env = deepcopy(self.env)
            val_env.reset(price_data=val_price_data)

        for episode in tqdm(range(n_episodes)):
            # reset environment
            state = self.env.reset(
                random_startpoint=random_startpoint, start_amount=start_amount
            )

            if self.epsilon_decay:
                self.epsilon = epsilon_start * epsilon_decay_step**episode

            # play until episode is terminated
            terminated = False
            while not terminated:
                action = self.make_decision(state, policy)
                next_state, reward, terminated, *_ = self.env.step(action)
                self.update_Q_table(state, action, reward, next_state)
                state = next_state

            # store episode data
            self.train_reward.append(self.env.episode_data.total_reward)

            # if (episode + 1) % 100 == 0:
            #    self.env.episode_data.plot()

            if val_env is not None:
                val_env.reset()
                terminated = False
                while not terminated:
                    action = self.make_decision(state, "greedy")
                    state, reward, terminated, *_ = val_env.step(action)

                self.val_reward.append(val_env.episode_data.total_reward)

    def validate(
        self,
        price_data: dict[datetime, float],
        random_startpoint: bool = False,
        start_amount: float = 0.5,
    ):
        # reset environment
        state = self.env.reset(
            price_data=price_data,
            random_startpoint=random_startpoint,
            start_amount=start_amount,
        )

        # play until episode is terminated
        terminated = False
        while not terminated:
            action = self.make_decision(state, "greedy")
            next_state, _, terminated, *_ = self.env.step(action)
            state = next_state

        return self.env.episode_data

    def plot_rewards_over_episode(self):
        plt.plot(self.train_reward, label="Train")
        plt.plot(self.val_reward, label="Validation")
        plt.legend()
        plt.title("Total reward over episode")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.show()

    def visualize_Q_table(self):
        # plot V value ~ price + hour
        # obs space is hour, price, res_level, action
        # x = price (20 bins)
        x = np.arange(self.env.observation_space.nvec[1])
        # y = time (24 bins)
        y = np.arange(self.env.observation_space.nvec[0])
        x, y = np.meshgrid(x, y)
        # z = V value
        # average out unnecessary dimensions
        z = np.mean(self.Qtable, axis=2)
        # max over actions ( = V value)
        z = np.max(z, axis=2)
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z)
        ax.set_xlabel("Price")
        ax.set_ylabel("Time")
        ax.set_zlabel("V value")
        plt.set_cmap("viridis")
        plt.show()

        # plot V value ~ price + res_level
        x = np.arange(self.env.observation_space.nvec[1])
        y = np.arange(self.env.observation_space.nvec[2])
        x, y = np.meshgrid(x, y)
        z = np.mean(self.Qtable, axis=0)
        z = np.max(z, axis=2).T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z)
        ax.set_xlabel("Price")
        ax.set_ylabel("Reservoir level")
        ax.set_zlabel("V value")
        plt.set_cmap("viridis")
        plt.show()

    def save(self, path: str):
        np.save(path, self.Qtable)

    def load(self, path: str):
        self.Qtable = np.load(path)


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        if torch is None:
            raise ImportError("PyTorch is not installed.")

        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.softmax(x)
        return x


class PolicyGradientAgent:
    def __init__(self, learning_rate: float, env: gym.Env):
        if torch is None:
            raise ImportError("PyTorch is not installed.")

        self.env = env
        self.state_size = len(env.observation_space.nvec)
        self.action_size = env.action_space.n

        self.policy_network = PolicyNetwork(self.state_size, self.action_size).to(
            DEVICE
        )
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=learning_rate
        )

    def get_action(self, state):
        state = torch.tensor(state).to(DEVICE).float().unsqueeze(0)
        probs = self.policy_network(state)
        action = torch.multinomial(probs, 1)

        return action.item()

    def update_policy(self, rewards, log_probs):
        returns = log_probs * rewards.sum()
        policy_loss = -returns.mean()
        policy_loss.requires_grad = True
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, n_episodes):
        for _ in tqdm(range(n_episodes)):
            state = self.env.reset()
            log_probs = []
            rewards = []
            terminated = False
            i = 0

            while not terminated:
                i += 1

                action = self.get_action(state)
                next_state, reward, terminated, *_ = self.env.step(action)
                log_probs.append(
                    torch.log(
                        self.policy_network(
                            torch.tensor(state).to(DEVICE).float().unsqueeze(0)
                        )[0, action]
                    )
                )
                rewards.append(reward)
                state = next_state

            self.update_policy(torch.tensor(rewards), torch.tensor(log_probs))

        return self.env.episode_data

    def validate(self, price_data: dict[datetime, float]):
        state = self.env.reset(price_data=price_data)
        terminated = False

        while not terminated:
            action = self.get_action(state)
            next_state, _, terminated, *_ = self.env.step(action)
            state = next_state

        return self.env.episode_data
