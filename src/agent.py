import gymnasium as gym
import numpy as np
from tqdm import tqdm
import random

# create agent
class Agent:
    def __init__(self, env: gym.Env, discount_factor: float = 0.99):
        self.env = env
        self.discount_factor = discount_factor

        # create Q table
        self.Qtable = np.zeros(
            np.append(self.env.observation_space.nvec, self.env.action_space.n)
        )

        print(f'self.Qtable.shape = {self.Qtable.shape}')

    def update_Q_table(self, state, action, reward, next_state):
        # update Q table
        current_idx = *state, action

        self.Qtable[current_idx] = (1 - self.alpha) * self.Qtable[current_idx] + self.alpha * (
            reward + self.discount_factor * self.Qtable[next_state].max()
        )

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
            action = np.random.choice(np.flatnonzero(self.Qtable[state] ==
                                                     self.Qtable[state].max()))
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
            state = self.env.reset(start_amount=0.0, random_startpoint=random_startpoint)  # to force the agent to fill the reservoir

            if self.epsilon_decay:
                self.epsilon = epsilon_start * epsilon_decay ** episode
            print(self.epsilon)

            terminated = False
            while not terminated:
                action = self.make_decision(state, policy)
                next_state, reward, terminated, *_ = self.env.step(action)
                self.update_Q_table(state, action, reward, next_state)
                state = next_state

            # store average reward
            self.episode_data.append(self.env.episode_data)

            # print progress

            if (episode + 1) % 100 == 0:
                self.env.episode_data.plot()

        return self.episode_data
 