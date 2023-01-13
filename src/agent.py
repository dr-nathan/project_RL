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

        # create Q table. Shape is hour, price, storage, action
        self.Qtable = np.zeros((self.env.observation_space[0].n,
                                self.env.observation_space[1].n+1,
                                self.env.observation_space[2].n+1,
                                self.env.action_space.n))

        print(f'self.Qtable.shape = {self.Qtable.shape}')

    def update_Q_table(self, state, action, reward, next_state):
        # update Q table
        self.Qtable[state[0], state[1], state[2], action] = (
            1 - self.alpha
        ) * self.Qtable[state[0], state[1], state[2], action] + self.alpha * (
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
            # to make sure we don't default to action 0
            action = np.random.choice(np.flatnonzero(self.Qtable[state[0], state[1], state[2]] ==
                                                     self.Qtable[state[0], state[1], state[2]].max()))
        return action

    def train(self, policy, n_episodes: int, epsilon: float = 0.1,
              epsilon_decay = False, alpha: float = 0.1, random_startpoint: bool = False):

        # intitialize stuff
        self.epsilon_decay = epsilon_decay
        self.episode_data = []
        if epsilon_decay:
            epsilon_start = 1
            epsilon_end = 0.01
            epsilon_decay = np.exp(np.log(epsilon_end / epsilon_start) / n_episodes)
        else:
            self.epsilon = epsilon

        self.alpha = alpha

        for episode in tqdm(range(n_episodes)):

            # reset environment
            state = self.env.reset(half_or_empty="empty", random_startpoint=random_startpoint)  # to force the agent to fill the reservoir

            if self.epsilon_decay:
                self.epsilon = epsilon_start * epsilon_decay ** episode
            print(self.epsilon)

            terminated = False
            while not terminated:
                action = self.make_decision(state, policy)
                next_state, reward, terminated, info = self.env.step(action)
                self.update_Q_table(state, action, reward, next_state)
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
    epsilon_decay = True
    epsilon = 0.5  # overriden if epsilon_decay is True
    alpha = 0.3
    n_episodes = 1000
    random_startpoint = True

    episode_data = agent.train("epsilon_greedy", n_episodes, epsilon, epsilon_decay, alpha, random_startpoint)

    print(agent.Qtable)
