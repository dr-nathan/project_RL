import copy
from collections import deque
from dataclasses import dataclass, field
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import random
from src.utils import cumsum, plt_col


class DQN(nn.Module):
    
    def __init__(self, env, learning_rate, seed, agent):

        super().__init__()
        self.seed = torch.manual_seed(seed)
        # NV: get the input features selected by agent
        input_features, *_ = agent.augment_state(env.state).shape
        action_space = env.discrete_action_space.n
        
        self.dense1 = nn.Linear(in_features=input_features, out_features=32)
        # self.dense2 = nn.Linear(in_features=128, out_features=64)
        # self.dense3 = nn.Linear(in_features=64, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=action_space)
        
        # Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):

        x = torch.relu(self.dense1(x))
        # x = torch.relu(self.dense2(x))
        # x = torch.relu(self.dense3(x))
        x = self.dense4(x)
        
        return x


class ExperienceReplay:
    
    def __init__(self, buffer_size, min_replay_size, agent, seed):

        """
        Params:
        env = environment that the agent needs to play
        buffer_size = max number of transitions that the experience replay buffer can store
        min_replay_size = min number of (random) transitions that the replay buffer needs to have when initialized
        seed = seed for random number generator for reproducibility
        """

        self.agent = agent
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0], maxlen=100)  # total episode rewards

        self.fill_replay_memory()

    def fill_replay_memory(self):
        """
        Fills the replay memory with random transitions.
        """

        # reset is now done from the agent. Simply deepcopies the initial env
        state = self.agent.reset_env()

        for i in range(self.min_replay_size):

            action = self.agent.choose_action(state, policy='random')  # choose random action, no NN yet
            next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
            transition = (state, action, reward, terminated, next_state)
            self.replay_buffer.append(transition)
            state = next_state

            if terminated or truncated:
                state = self.agent.reset_env()
            
    def sample(self, batch_size):

        # sample random transitions from the replay memory
        transitions = random.sample(self.replay_buffer, batch_size)

        # convert to array where needed, then to tensor (faster than directly to tensor)
        observations = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rewards = [t[2] for t in transitions]
        dones = np.asarray([t[3] for t in transitions])
        new_observations = [t[4] for t in transitions]

        # preprocess observations (normalize, select features)
        observations = np.asarray([self.agent.preprocess_state(obs) for obs in observations])
        new_observations = np.asarray([self.agent.preprocess_state(obs) for obs in new_observations])
        # rewards = np.asarray((rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8))  # normalize rewards

        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, dones_t, new_observations_t


class DDQNAgent:
    
    def __init__(self, env, val_env, device, epsilon, epsilon_decay, epsilon_start,
                 epsilon_end, n_episodes, discount_rate, lr, buffer_size, seed):
        """
        Params:
        env = name of the environment that the agent needs to play
        device = set up to run CUDA operations
        epsilon_decay = Decay period until epsilon start -> epsilon end
        epsilon_start = starting value for the epsilon value
        epsilon_end = ending value for the epsilon value
        discount_rate = discount rate for future rewards
        lr = learning rate
        buffer_size = max number of transitions that the experience replay buffer can store
        seed = seed for random number generator for reproducibility
        """

        self.original_env = env
        self.env = copy.deepcopy(env)
        self.val_env = val_env
        self.seed = seed
        self.device = device
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.n_episodes = n_episodes
        self.episode_data = DamEpisodeData()

        self.DEBUG = False

        if epsilon_decay:
            self.epsilon = epsilon_start
            self.epsilon_decay_step = np.exp(np.log(epsilon_end / epsilon_start) / self.n_episodes)
        else:
            self.epsilon = epsilon
            self.epsilon_decay_step = 1.0
        
        self.replay_memory = ExperienceReplay(self.buffer_size, min_replay_size=10000,
                                              agent=self, seed=self.seed)
        self.online_network = DQN(self.env, self.learning_rate, seed=self.seed, agent=self).to(self.device)

        self.target_network = DQN(self.env, self.learning_rate, seed=self.seed, agent=self).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def training_loop(self, batch_size):

        # reset the environment
        state = self.reset_env()

        train_rewards = []
        val_rewards = []

        epsilons = []

        for iteration in tqdm(range(self.n_episodes)):

            # play one move, add the transition to the replay memory
            state, action, reward = self.play_action(state)

            # decay epsilon
            self.epsilon *= self.epsilon_decay_step
            epsilons.append(self.epsilon)

            if (iteration+1) % 4 == 0:
                # sample a batch of transitions from the replay memory, and update online network
                self.learn(batch_size=batch_size)

            # every 500 iterations, update the target network
            if (iteration+1) % 500 == 0:
                self.target_network.load_state_dict(self.online_network.state_dict())

            # get some statistics every time the agent has seen the whole dataset
            if (iteration+1) % self.env.len == 0:
                data_train, _ = self.validate(copy.deepcopy(self.original_env))
                data_val, _ = self.validate(copy.deepcopy(self.val_env))
                train_rewards.append(data_train)
                val_rewards.append(data_val)

            state = self.env.state
            self.episode_data.add(
                datetime.datetime(int(state[6]), 1, 1) +  # year
                datetime.timedelta(days=int(state[4]) - 1, hours=int(state[2])),  # day of the year + hour
                state[0],
                action,
                action * self.env.max_flow,
                state[1],
                reward
            )

        plot_rewards(train_rewards, val_rewards)
        plot_nn_weights(self.online_network)
        if self.DEBUG:
            plt.plot(epsilons)
            plt.show()

        self.episode_data.debug_plot()

    def play_action(self, state):

        action = self.choose_action(state, "epsilon_greedy")
        next_state, reward, terminated, _, _ = self.env.step(action)
        self.replay_memory.replay_buffer.append((state, action, reward, terminated, next_state))

        state = next_state

        if terminated:

            state = self.reset_env()

        return state, action, reward

    def choose_action(self, state, policy):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if policy == "random":
            action = self.env.discrete_action_space.sample()
            return self.encode_decode_actions(action, "decode")  # convert to -1, 0, 1
        elif policy == "greedy":
            state = self.preprocess_state(state)
            action = self.online_network.forward(state).argmax().item()
            return self.encode_decode_actions(action, "decode")
        elif policy == "epsilon_greedy":
            if random.random() < self.epsilon:
                action = self.env.discrete_action_space.sample()
                return self.encode_decode_actions(action, "decode")
            else:
                state = self.preprocess_state(state)
                action = self.online_network.forward(state).argmax().item()
                return self.encode_decode_actions(action, "decode")
        else:
            raise ValueError("Unknown policy")

    def learn(self, batch_size):

        """
        Params:
        batch_size = number of transitions that will be sampled
        """

        # sampler also takes care of normalizing the features + rewards and converting to tensors
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        target_q_values = self.target_network.forward(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values

        # Compute loss
        q_values = self.online_network.forward(observations_t)
        actions_t = self.encode_decode_actions(actions_t, "encode")
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t.unsqueeze(-1))
        # loss = F.mse_loss(action_q_values, targets)
        loss = f.smooth_l1_loss(action_q_values, targets)

        # Gradient descent to update the weights of the neural network
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

    def validate(self, env):
        # assumes that the environment is reset
        self.episode_data = DamEpisodeData()
        state, *_ = env.step(0)

        # play until episode is terminated
        total_reward = 0
        terminated = False
        while not terminated:
            action = self.choose_action(state, "greedy")
            next_state, reward, terminated, _, *_ = env.step(action)
            total_reward += reward
            self.episode_data.add(
                datetime.datetime(int(state[6]), 1, 1) +  # year
                datetime.timedelta(days=int(state[4]), hours=int(state[2])),  # day of the year + hour
                state[0],
                action,
                action * env.max_flow,
                state[1],
                reward
            )
            state = next_state
        if self.DEBUG:
            self.episode_data.debug_plot()

        return total_reward, self.episode_data

    def reset_env(self):

        self.env = copy.deepcopy(self.original_env)
        self.env.reset(seed=self.seed)  # TODO: check if necessary
        self.episode_data = DamEpisodeData()
        # make fake action to get the first state
        state, _, _, _, _ = self.env.step(0)

        return state

    @staticmethod
    def encode_decode_actions(action, direction):
        if direction == "encode":
            if any(a == 2 for a in action):
                raise ValueError("Action should be -1, 0 or 1")
            return torch.as_tensor(np.asarray([1 if a == -1 else 2 if a == 1 else 0 for a in action]))

        elif direction == "decode":
            if action == -1:
                raise ValueError("Action should be 0, 1 or 2")
            return -1 if action == 1 else 1 if action == 2 else 0
        # TODO: find a fix for this, ew

    def preprocess_state(self, state):
        state[0] = state[0] / self.env.max_volume
        state[1] = state[1] / 200
        state[2] = state[2] / 24
        state[3] = state[3] / 7
        state[4] = state[4] / 365
        state[5] = state[5] / 12

        state = self.augment_state(state)

        return state

    @staticmethod
    def augment_state(state):
        # only keep 3 first features
        state = state[:3]
        # TODO: add the other features

        return state


def plot_rewards(train_rewards, val_rewards):
    plt.plot(train_rewards, label="train")
    plt.plot(val_rewards, label="val")
    plt.legend()
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("Reward over time")
    plt.show()


# to get feature importance
def plot_nn_weights(model):
    weights = model.dense1.weight.detach().numpy()
    # get absolute mean of weights
    weights = np.abs(weights).mean(axis=0)
    plt.bar(range(len(weights)), weights)
    plt.xlabel("feature")
    plt.ylabel("weight")
    plt.title("Feature importance")
    plt.show()


@dataclass
class DamEpisodeData:
    """Dataclass to store episode data for a dam environment"""

    date: list[datetime.datetime] = field(default_factory=list)
    storage: list[float] = field(default_factory=list)
    action: list[float] = field(default_factory=list)
    flow: list[float] = field(default_factory=list)
    price: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    reward_cumulative = property(lambda self: cumsum(self.reward))
    total_reward = property(lambda self: sum(self.reward))

    def __len__(self):
        return len(self.date)

    def add(
        self,
        date: datetime.datetime,
        storage: float,
        action: float,
        flow: float,
        price: float,
        reward: float,
    ):
        self.date.append(date)
        self.storage.append(storage)
        self.action.append(action)
        self.flow.append(flow)
        self.price.append(price)
        self.reward.append(reward)

    def debug_plot(self, title: str | None = None):
        sns.set()
        fig, axs = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

        if title:
            fig.suptitle(title)

        axs[0].plot(self.date, self.storage)
        axs[0].set_title("Storage")

        axs[1].scatter(self.date, self.action, s=1, marker="x")
        axs[1].set_title("Action")

        axs[2].plot(self.date, self.flow)
        axs[2].set_title("Flow")

        axs[3].plot(self.date, self.price)
        axs[3].set_title("Price")

        axs[4].plot(self.date, self.reward)
        axs[4].set_title("Reward")

        axs[5].plot(self.date, self.reward_cumulative)
        axs[5].set_title("Cumulative reward")

        fig.tight_layout()
        plt.show()

    def plot_fancy(self):
        sns.set()
        price = self.price[-1001:-1]
        action = self.action[-1000:]
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        cols = plt_col(action)

        df = pd.DataFrame({"price": price, "action": action}).reset_index()
        df.action = df.action.map({0: "nothing", 1: "sell", 2: "buy"})

        sns.scatterplot(
            data=df,
            x="index",
            y="price",
            hue="action",
            palette={"nothing": "blue", "sell": "green", "buy": "red"},
        )
        plt.ylim(0, 170)
        plt.title("Action on the prices over time")

        # axs.scatter(range(len(price)),price,s=100, c=cols,marker= 'o', label=cols)
        # axs.legend()
        # axs.set_title("Action on the prices")
        # fig.tight_layout()
        plt.show()
