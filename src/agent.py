import random
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# create agent
class Agent:
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
            action = np.argmax(self.Qtable[state])
        elif policy == "epsilon_greedy":
            action = self.choose_action_eps_greedy(state)
        else:
            raise ValueError("Unknown policy")

        return action

    def choose_action_eps_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # to make sure we don't default to action 0
            action = np.random.choice(
                np.flatnonzero(self.Qtable[state] == self.Qtable[state].max())
            )
        return action

    def train(
        self,
        policy,
        n_episodes: int,
        epsilon: float = 0.1,
        epsilon_decay: bool = False,
        alpha: float = 0.1,
        random_startpoint: bool = False,
        start_amount: float = 0.5,
    ):

        # intitialize stuff
        self.epsilon_decay = epsilon_decay
        self.tot_reward = []

        if self.epsilon_decay:
            epsilon_start = 1
            epsilon_end = 0.01
            epsilon_decay_step = np.exp(
                np.log(epsilon_end / epsilon_start) / n_episodes
            )
        else:
            self.epsilon = epsilon

        self.alpha = alpha

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
            self.tot_reward.append(self.env.episode_data.total_reward)

            # if (episode + 1) % 100 == 0:
            #    self.env.episode_data.plot()

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

    def plot_rewards_over_episode(self):
        plt.plot(self.tot_reward)
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

