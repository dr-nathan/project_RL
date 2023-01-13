from environment import DiscreteDamEnv
from utils import convert_dataframe

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm


# create agent
class Agent:
    def __init__(self, env: gym.Env, discount_factor: float = 0.99):
        self.env = env
        self.discount_factor = discount_factor

        # create Q table
        self.Qtable = np.zeros((self.env.observation_space[0].n,
                                self.env.observation_space[1].n+1,
                                self.env.observation_space[2].n,
                                self.env.action_space.n))
        print(f'self.Qtable.shape = {self.Qtable.shape}')

    def update_Q_table(self, state, action, reward, next_state, alpha: float = 0.1):
        # update Q table
        self.Qtable[state[0], state[1], state[2], action] = (
            1 - alpha
        ) * self.Qtable[state[0], state[1], state[2], action] + alpha * (
            reward + self.discount_factor * np.max(self.Qtable[next_state[0], next_state[1], next_state[2]])
        )

    def make_decision(self, state, policy: str = "epsilon_greedy"):
        if policy == "greedy":
            action = np.argmax(self.Qtable[state[0], state[1], state[2]])
        elif policy == "epsilon_greedy":
            action = self.choose_action_eps_greedy(state)
        else:
            raise ValueError("Unknown policy")

        return action

    def choose_action_eps_greedy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Qtable[state[0], state[1], state[2]])
        return action

    def train(self, policy, n_episodes: int, epsilon: float = 0.1, alpha: float = 0.1):

        # intitialize stuff
        self.episode_data = []
        self.epsilon = epsilon
        self.alpha = alpha

        for episode in tqdm(range(n_episodes)):

            # reset environment
            state = self.env.reset()

            # pick random startpoint
            # state = self.env.pick_random_startpoint()

            terminated = False
            while not terminated:
                action = self.make_decision(state, policy)
                next_state, reward, terminated, info = self.env.step(action)
                self.update_Q_table(state, action, reward, next_state, self.alpha)
                state = next_state

            # store average reward
            self.episode_data.append(self.env.episode_data)

            # print progress

            if (episode + 1) % 100 == 0:
                self.env.episode_data.plot()

        return self.episode_data


if __name__ == '__main__':

    # load data
    train_data = pd.read_excel("../data/train.xlsx")
    train_data = convert_dataframe(train_data)

    # create environment and agent
    environment = DiscreteDamEnv(train_data)
    agent = Agent(environment)

    # train agent
    episode_data = agent.train("epsilon_greedy", 1000, epsilon=0.4, alpha=0.3)

    print(agent.Qtable)
