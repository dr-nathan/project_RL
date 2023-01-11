import gymnasium as gym
import numpy as np

from environment import DiscreteDamEnv
from utils import convert_dataframe

import pandas as pd

# load data
train_data = pd.read_excel("../data/train.xlsx")
train_data = convert_dataframe(train_data)

environment = DiscreteDamEnv(train_data)

# create agent
class Agent:
    def __init__(self, env: gym.Env, discount_factor: float = 0.99):
        self.env = env
        self.discount_factor = discount_factor

        self.action_space = self.env.action_space.n  # TODO: n?

        self.observation_space = (self.env.observation_space[0].n, self.env.observation_space[1].n)

    def create_Q_table(self):

        self.Qtable = np.zeros((self.observation_space[0], self.observation_space[1], self.action_space))

    def make_decision(self, state, policy: str = "greedy"):
        if policy == "greedy":
            action = np.argmax(self.Qtable[state[0], state[1]])
        elif policy == "epsilon_greedy":
            action = self.choose_action_eps_greedy(state, self.epsilon)

        return action

    def choose_action_eps_greedy(self, state, epsilon: float = 0.1):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.Qtable[state[0], state[1]])
        return action

    def train(self, policy, n_episodes: int, epsilon: float = 0.1, alpha: float = 0.1):

        # intitialize stuff
        self.rewards = []
        self.average_rewards = []
        self.epsilon = epsilon
        self.alpha = alpha

        # create Q table
        self.create_Q_table()

        for episode in range(n_episodes):

            # reset environment
            self.env.reset()
            state = self.env._get_state()

            done = False
            while not done:
                action = self.make_decision(state, policy=policy)
                next_state, reward = self.env.step(action)
                self.update_Q_table(state, action, reward, next_state)
                state = next_state


        pass


agent = Agent(environment)

agent.train(policy="epsilon_greedy", n_episodes=1000)