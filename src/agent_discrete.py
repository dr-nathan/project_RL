import random
from copy import deepcopy
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.environment import DiscreteDamEnv


class QLearnAgent:
    def __init__(self, env: DiscreteDamEnv, discount_factor: float = 0.98):
        self.env = env
        self.discount_factor = discount_factor

        # create Q table
        self.Qtable = (
            np.ones(np.append(self.env.observation_space.nvec, self.env.action_space.n))
            * 1000
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
        val_price_data: dict[datetime, float] | None = None,
    ):

        # intitialize stuff
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.train_reward = []
        self.val_reward = []

        if self.epsilon_decay:
            epsilon_start = 1
            epsilon_end = 0.1
            epsilon_decay_step = np.exp(
                np.log(epsilon_end / epsilon_start) / n_episodes
            )
        else:
            self.epsilon = epsilon

        val_env = None
        if val_price_data is not None:
            val_env = deepcopy(self.env)
            val_env.reset(price_data=val_price_data)

        for episode in tqdm(range(n_episodes)):
            # reset environment
            state,_ = self.env.reset(
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
        state,_ = self.env.reset(
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

    def save(self, path: str):
        np.save(path, self.Qtable)

    def load(self, path: str):
        self.Qtable = np.load(path)

    def plot_rewards_over_episode(self):
        plt.plot(self.train_reward, label="Train")
        plt.plot(self.val_reward, label="Validation")
        plt.legend()
        plt.title("Total reward over episode")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.show()

    def plot_price_bins(self, train_price_data, val_price_data):
        sns.set()
        # cap prices
        train_price_data = [min(170, p) for p in train_price_data.values()]
        val_price_data = [min(170, p) for p in val_price_data.values()]
        # make long df (sucks, but necessary for seaborn).
        df = pd.DataFrame(
            {
                "Price": train_price_data + val_price_data,
                "Set": ["Train"] * len(train_price_data)
                + ["Validation"] * len(val_price_data),
            }
        )
        # plot
        sns.displot(df, x="Price", hue="Set", kind="kde", fill=True)
        plt.axvline(self.env.quantiles[0], color="red", lw=0.7, label='quantiles')
        for q in self.env.quantiles[1:]:
            plt.axvline(q, color="red", lw=0.7)
        plt.legend()
        plt.title("Price distribution")
        plt.xlabel("Price")
        plt.tight_layout()
        plt.show()

    def plot_price_distribution(self):
        sns.set()
        # make dist plot
        sns.displot(self.env.price_data.values(), kde=True)
        plt.title("Price distribution")
        plt.xlabel("Price")
        plt.ylabel("Count")
        # mark quantiles
        for q in [0.4, 0.6]:
            plt.axvline(np.quantile(self.env.price_data.values(), q), color="red")
        plt.xlim(0, 170)
        plt.tight_layout()
        plt.show()

    def visualize_Q_table(self):

        ## 3D plots ##
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
        # get argamx over action dimension
        z_action = np.argmax(z, axis=2)
        # get V value
        z = np.max(z, axis=2)
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = {0: "orange", 1: "green", 2: "red"}
        z_action = np.vectorize(colors.get)(z_action)
        ax.plot_surface(x, y, z, facecolors=z_action)
        # set labels for colors
        labels = ["Hold", "Sell", "Buy"]
        handles = [
            mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))
        ]
        ax.legend(handles=handles)
        ax.set_xlabel("Price")
        ax.set_ylabel("Time")
        ax.set_zlabel("V value")
        plt.show()

        # plot V value ~ price + res_level
        x = np.arange(self.env.observation_space.nvec[1])
        y = np.arange(self.env.observation_space.nvec[2])
        x, y = np.meshgrid(x, y)
        z = np.mean(self.Qtable, axis=0)
        z_action = np.argmax(z, axis=2).T
        z_action = np.vectorize(colors.get)(z_action)
        z = np.max(z, axis=2).T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, facecolors=z_action)
        labels = ["Hold", "Sell", "Buy"]
        handles = [
            mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))
        ]
        ax.legend(handles=handles)
        ax.set_xlabel("Price")
        ax.set_ylabel("Reservoir level")
        ax.set_zlabel("V value")
        plt.set_cmap("viridis")
        plt.show()

        ## 2D plots ##
        # plot V value ~ price
        x = np.arange(self.env.observation_space.nvec[1])
        y = np.mean(self.Qtable, axis=(0, 2))
        y = np.max(y, axis=1)
        plt.plot(x, y)
        plt.title("V value ~ price")
        plt.xlabel("Price")
        plt.ylabel("V value")
        plt.show()

        # plot V value ~ res_level
        x = np.arange(self.env.observation_space.nvec[2])
        y = np.mean(self.Qtable, axis=(0, 1))
        y = np.max(y, axis=1)
        plt.plot(x, y)
        plt.title("V value ~ reservoir level")
        plt.xlabel("Reservoir level")
        plt.ylabel("V value")
        plt.show()

        # plot V value ~ time
        x = np.arange(self.env.observation_space.nvec[0])
        y = np.mean(self.Qtable, axis=(1, 2))
        y = np.max(y, axis=1)
        plt.plot(x, y)
        plt.title("V value ~ time")
        plt.xlabel("Time")
        plt.ylabel("V value")
        plt.show()
