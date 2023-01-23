import gymnasium as gym
import numpy as np
from tqdm import tqdm


# create agent
class Agent:
    def __init__(self, env: gym.Env, discount_factor: float = 0.99):
        self.env = env
        self.discount_factor = discount_factor

        # create Q table
        self.Qtable = np.zeros(
            np.append(self.env.observation_space.nvec, self.env.action_space.n)
        )

    def update_Q_table(self, state, action, reward, next_state, alpha: float = 0.1):
        current_idx = *state, action

        self.Qtable[current_idx] = (1 - alpha) * self.Qtable[current_idx] + alpha * (
            reward + self.discount_factor * self.Qtable[next_state].max()
        )

    def make_decision(self, state, policy: str = "epsilon_greedy"):
        if policy == "greedy":
<<<<<<< Updated upstream
            action = np.argmax(self.Qtable[state])
        elif policy == "epsilon_greedy":
            action = self.choose_action_eps_greedy(state, self.epsilon)
        else:
            raise ValueError("Unknown policy")
=======
            return np.argmax(self.Qtable[state])

        if policy == "epsilon_greedy":
            return self.choose_action_eps_greedy(state)

            raise ValueError("Unknown policy")

    def choose_action_eps_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
>>>>>>> Stashed changes

        return action

    def choose_action_eps_greedy(self, state, epsilon: float = 0.1):
        if np.random.uniform(0, 1) < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Qtable[state])
        return action

    def train(self, policy, n_episodes: int, epsilon: float = 0.1, alpha: float = 0.1):

        # intitialize stuff
        self.episode_data = []
        self.epsilon = epsilon
        self.alpha = alpha

        for episode in tqdm(range(n_episodes)):
<<<<<<< Updated upstream
=======

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

    def save(self, path: str):
        np.save(path, self.Qtable)

    def load(self, path: str):
        self.Qtable = np.load(path)
>>>>>>> Stashed changes

            # reset environment
            state = self.env.reset()

            # pick random startpoint
            # state = self.env.pick_random_startpoint()

            terminated = False
            while not terminated:
                action = self.make_decision(state, policy=policy)
                next_state, reward, terminated, info = self.env.step(action)
                self.update_Q_table(state, action, reward, next_state, self.alpha)
                state = next_state

            # store average reward
            self.episode_data.append(self.env.episode_data)

            # print progress

            if (episode + 1) % 100 == 0:
                self.env.episode_data.plot()

        return self.episode_data
