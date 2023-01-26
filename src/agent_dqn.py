from collections import deque

import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class DQN(nn.Module):
    
    def __init__(self, env, learning_rate):

        super().__init__()
        input_features, *_ = env.observation_space.shape
        action_space = env.action_space.n
        
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
    
    def __init__(self, env, buffer_size, min_replay_size, agent):

        """
        Params:
        env = environment that the agent needs to play
        buffer_size = max number of transitions that the experience replay buffer can store
        min_replay_size = min number of (random) transitions that the replay buffer needs to have when initialized
        seed = seed for random number generator for reproducibility
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0], maxlen=100)  # total episode rewards

        self.fill_replay_memory(agent)

    def fill_replay_memory(self, agent):
        """
        Fills the replay memory with random transitions.
        """

        state = self.env.reset()

        for i in range(self.min_replay_size):

            action = agent.choose_action(state, policy='epsilon_greedy')
            next_state, reward, terminated, _ = self.env.step(action)
            transition = (state, action, reward, terminated, next_state)
            self.replay_buffer.append(transition)
            state = next_state

            if terminated:
                state = self.env.reset()
            
    def sample(self, batch_size):

        # sample random transitions from the replay memory
        transitions = random.sample(self.replay_buffer, batch_size)

        observations = [t[0] for t in transitions]
        actions = [t[1] for t in transitions]
        rewards = [t[2] for t in transitions]
        dones = [t[3] for t in transitions]
        new_observations = [t[4] for t in transitions]

        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, dones_t, new_observations_t


class DDQNAgent:
    
    def __init__(self, env, device, epsilon, epsilon_decay, epsilon_start,
                 epsilon_end, n_episodes, discount_rate, lr, buffer_size):
        '''
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
        '''

        self.env = env
        self.device = device
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.n_episodes = n_episodes

        if epsilon_decay:
            self.epsilon = epsilon_start
            self.epsilon_decay_step = np.exp(np.log(epsilon_end / epsilon_start) / self.n_episodes)
        else:
            self.epsilon = epsilon
            self.epsilon_decay_step = 1.0
        
        self.replay_memory = ExperienceReplay(self.env, self.buffer_size, min_replay_size=1000, agent=self)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)

        self.target_network = DQN(self.env, self.learning_rate).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def training_loop(self, batch_size):

        """
        Params:
        env = name of the environment that the agent needs to play
        agent= which agent is used to train
        max_episodes = maximum number of games played
        target = boolean variable indicating if a target network is used (this will be clear later)
        seed = seed for random number generator for reproducibility

        Returns:
        average_reward_list = a list of averaged rewards over 100 episodes of playing the game
        """

        for iteration in tqdm(range(self.n_episodes)):

            # reset the environment
            state = self.env.reset()

            # play the game, add the transition to the replay memory
            self.play(state, batch_size=batch_size)

            # decay epsilon
            self.epsilon *= self.epsilon_decay_step

            self.learn(batch_size=batch_size)

            if (iteration+1) % 100 == 0:

                self.update_target_network()

            if (iteration+1) % 10000 == 0:
                self.env.episode_data.debug_plot()

        return self.env.episode_data

    def learn(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        '''
        
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        target_q_values = self.target_network.forward(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1-dones_t) * max_target_q_values

        # Compute loss
        q_values = self.online_network.forward(observations_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        # loss = F.mse_loss(action_q_values, targets)
        loss = F.smooth_l1_loss(action_q_values, targets)
        
        # Gradient descent to update the weights of the neural network
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()
        
    def update_target_network(self):

        self.target_network.load_state_dict(self.online_network.state_dict())

    def choose_action(self, state, policy):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if policy == "random":
            return self.env.action_space.sample()
        elif policy == "greedy":
            return self.online_network.forward(state).argmax().item()
        elif policy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                return self.online_network.forward(state).argmax().item()
        else:
            raise ValueError("Unknown policy")

    def play(self, state, batch_size):

        for i in range(batch_size):
            action = self.choose_action(state, "epsilon_greedy")
            next_state, reward, terminated, *_ = self.env.step(action)
            self.replay_memory.replay_buffer.append((state, action, reward, terminated, next_state))
            state = next_state

            if terminated:
                state = self.env.reset()

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
            action = self.choose_action(state, "greedy")
            next_state, _, terminated, *_ = self.env.step(action)
            state = next_state

        return self.env.episode_data  
