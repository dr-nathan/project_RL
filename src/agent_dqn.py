import copy
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm


class DQN(nn.Module):
    def __init__(self, env, learning_rate, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # NV: get the input features selected by agent
        input_features, *_ = env.state.shape
        action_space = env.discrete_action_space.n

        self.dense1 = nn.Linear(in_features=input_features, out_features=64)
        # self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=action_space)

        # Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        # x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = self.dense4(x)

        return x


class ExperienceReplay:
    def __init__(self, buffer_size: int, min_replay_size: int, agent: object, seed: int):

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

        state = self.agent.env.reset()

        for i in range(self.min_replay_size):

            action = self.agent.choose_action(
                state, policy="random"
            )  # choose random action, no NN yet
            next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
            transition = (state, action, reward, terminated, next_state)
            self.replay_buffer.append(transition)
            state = next_state

            if terminated or truncated:
                state = self.agent.reset_env()

    def sample(self, batch_size: int):

        # sample random transitions from the replay memory
        transitions = random.sample(self.replay_buffer, batch_size)

        # convert to array where needed, then to tensor (faster than directly to tensor)
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        # normalize rewards
        # rewards = np.asarray((rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8))

        observations_t = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(-1)
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        dones_t = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        new_observations_t = torch.as_tensor(
            new_observations, dtype=torch.float32, device=self.device
        )

        return observations_t, actions_t, rewards_t, dones_t, new_observations_t


class DDQNAgent:
    def __init__(
            self,
            env,
            val_env,
            device: torch.device,
            epsilon: float = 1.0,
            epsilon_decay: bool = True,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            n_episodes: int = 20,
            discount_rate: float = 0.99,
            lr: float = 5e-4,
            buffer_size: int = 100000,
            seed: int = 7,
    ):

        self.env = env
        self.val_env = val_env
        self.seed = seed
        self.device = device
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.n_episodes = n_episodes

        self.DEBUG = False

        if epsilon_decay:
            self.epsilon = epsilon_start
            self.epsilon_decay_step = np.exp(
                np.log(epsilon_end / epsilon_start) / self.n_episodes
            )
        else:
            self.epsilon = epsilon
            self.epsilon_decay_step = 1.0

        self.replay_memory = ExperienceReplay(
            self.buffer_size, min_replay_size=10000, agent=self, seed=self.seed
        )
        self.online_network = DQN(self.env, self.learning_rate, seed=self.seed).to(
            self.device
        )

        self.target_network = DQN(self.env, self.learning_rate, seed=self.seed).to(
            self.device
        )
        self.target_network.load_state_dict(self.online_network.state_dict())

    def training_loop(self, batch_size: int):

        # reset the environment
        state = self.env.reset()

        train_rewards = []
        val_rewards = []

        epsilons = []

        for iteration in tqdm(range(self.n_episodes)):

            # play one move, add the transition to the replay memory
            state, *_ = self.play_action(state)

            # decay epsilon
            self.epsilon *= self.epsilon_decay_step
            epsilons.append(self.epsilon)

            if (iteration + 1) % 4 == 0:
                # sample a batch of transitions from the replay memory, and update online network
                self.learn(batch_size=batch_size)

            # every 500 iterations, update the target network
            if (iteration + 1) % 500 == 0:
                self.target_network.load_state_dict(self.online_network.state_dict())

            # get some statistics every time the agent has seen the whole dataset
            if (iteration + 1) % len(self.env) == 0:
                data_train, _ = self.validate(copy.deepcopy(self.env))
                data_val, _ = self.validate(copy.deepcopy(self.val_env))
                train_rewards.append(data_train)
                val_rewards.append(data_val)

        plot_rewards(train_rewards, val_rewards)
        plot_nn_weights(self.online_network)
        if self.DEBUG:
            plt.plot(epsilons)
            plt.show()

    def play_action(self, state: np.ndarray):

        action = self.choose_action(state, "epsilon_greedy")
        next_state, reward, terminated, *_ = self.env.step(action)
        self.replay_memory.replay_buffer.append(
            (state, action, reward, terminated, next_state)
        )

        state = next_state

        if terminated:
            state = self.env.reset()

        return state, action, reward

    def choose_action(self, state: np.ndarray, policy: str):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if policy == "random":
            return self.env.discrete_action_space.sample()

        if policy == "greedy":
            return self.online_network.forward(state).argmax().item()

        if policy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return self.env.discrete_action_space.sample()

            return self.online_network.forward(state).argmax().item()

        raise ValueError("Unknown policy")

    def learn(self, batch_size: int):

        """
        Params:
        batch_size = number of transitions that will be sampled
        """

        # sampler also takes care of normalizing the features + rewards and converting to tensors
        (
            observations_t,
            actions_t,
            rewards_t,
            dones_t,
            new_observations_t,
        ) = self.replay_memory.sample(batch_size)

        # normalize rewards
        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        self.target_network.eval()
        target_q_values = self.target_network.forward(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        # normalize rewards
        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-9)

        targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values

        # Compute loss
        self.online_network.train()
        q_values = self.online_network.forward(observations_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        # loss = F.mse_loss(action_q_values, targets)
        loss = f.smooth_l1_loss(action_q_values, targets)

        # Gradient descent to update the weights of the neural network
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

    def validate(self, env):

        # reset the environment
        state = env.reset()

        # play until episode is terminated
        total_reward = 0
        terminated = False

        while not terminated:
            action = self.choose_action(state, "greedy")
            next_state, reward, terminated, *_ = env.step(action)
            total_reward += reward
            state = next_state
        if self.DEBUG:
            env.episode_data.debug_plot()

        return total_reward, env.episode_data

    def visualize_features(self):

        ## 3D plots ##
        # plot V value ~ price + hour
        # obs space is hour, price, res_level, action
        # x = price (20 bins)
        x = np.arange(self.env.observation_space.nvec[1])
        # y = time (24 bins)
        y = np.arange(self.env.observation_space.nvec[0])
        x, y = np.meshgrid(x, y)
        # z = V value
        # get all relevant instances in replay memory
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
        x = np.linspace(0, 1, self.env.observation_space.nvec[1])
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
    fig, ax = plt.subplots()
    ax.bar(range(len(weights)), weights)
    plt.xlabel("feature")
    plt.ylabel("weight")
    plt.title("Feature importance")
    plt.show()

