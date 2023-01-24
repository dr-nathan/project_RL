import gymnasium as gym
import numpy as np
from tqdm import tqdm

import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from copy import deepcopy




class DQN(nn.Module):
    
    def __init__(self, env, learning_rate):
        
        '''
        Params:
        env = environment that the agent needs to play
        learning_rate = learning rate used in the update
        
        '''
        
        super(DQN,self).__init__()
        input_features,*_ = env.observation_space.shape
        action_space = env.action_space.n
        

        
        self.dense1 = nn.Linear(in_features = input_features, out_features = 128)
        self.dense2 = nn.Linear(in_features = 128, out_features = 64)
        self.dense3 = nn.Linear(in_features = 64, out_features = 32)
        self.dense4 = nn.Linear(in_features = 32, out_features = action_space)
        
        #Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
    def forward(self, x):
        
        '''
        Params:
        x = observation
        '''
 
        
        #Solution:
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = self.dense4(x)
        
        return x
    


class ExperienceReplay:
    
    def __init__(self, env, buffer_size, min_replay_size = 1000, seed = 123):
        
        '''
        Params:
        env = environment that the agent needs to play
        buffer_size = max number of transitions that the experience replay buffer can store
        min_replay_size = min number of (random) transitions that the replay buffer needs to have when initialized
        seed = seed for random number generator for reproducibility
        '''
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0], maxlen = 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print('Please wait, the experience replay buffer will be filled with random transitions')
                
        obs = self.env.reset(seed=seed)
        for _ in range(self.min_replay_size):

            action = env.action_space.sample()
            new_obs, rew, terminated, * _ = env.step(action)
            done = terminated 

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
    
            if done:
                obs, _ = env.reset(seed=seed)
        
        print('Initialization with random transitions is done!')
      
          
    def add_data(self, data): 
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        self.replay_buffer.append(data)
            
    def sample(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        
        Returns:
        tensor of observations, actions, rewards, done (boolean) and next observation 
        '''
        
        transitions = random.sample(self.replay_buffer, batch_size)

        #Solution
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        #PyTorch needs these arrays as tensors!, don't forget to specify the device! (cpu / GPU)
        observations_t = torch.as_tensor(observations, dtype = torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype = torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype = torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype = torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype = torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, dones_t, new_observations_t
    
    def add_reward(self, reward):
        
        '''
        Params:
        reward = reward that the agent earned during an episode of a game
        '''
        
        self.reward_buffer.append(reward)
        




class DDQNAgent:
    
    def __init__(self, env_name, device, epsilon_decay, 
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed = 123):
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
        self.env = env_name
        #self.env = gym.make(self.env_name, render_mode = None)
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        
        self.replay_memory = ExperienceReplay(self.env, self.buffer_size, seed = seed)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)

        self.target_network = DQN(self.env, self.learning_rate).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
    def choose_action(self, observation , policy:str = 'epsilon_greedy'):
        
        '''
        Params:
        step = the specific step number 
        observation = observation input
        greedy = boolean that
        
        Returns:
        action: action chosen (either random or greedy)
        epsilon: the epsilon value that was used 
        '''
        
        #epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        epsilon = 0.8
        if policy == 'epsilon_greedy':
            random_sample = random.random()
    
            if (random_sample <= epsilon) :
            #Random action
                return self.env.action_space.sample()
        
        if policy == 'greedy':

            #Greedy action
            obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
            q_values = self.online_network(obs_t.unsqueeze(0))
        
            max_q_index = torch.argmax(q_values, dim = 1)[0]
            action = max_q_index.detach().item()
        
            return action
    
    
    def return_q_value(self, observation):
        '''
        Params:
        observation = input value of the state the agent is in
        
        Returns:
        maximum q value 
        '''
        #We will need this function later for plotting the 3D graph
        
        obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
        q_values = self.online_network(obs_t.unsqueeze(0))
        
        return torch.max(q_values).item()
        
    def learn(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        '''
        
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        #Compute targets, note that we use the same neural network to do both! This will be changed later!

        target_q_values = self.target_network(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1-dones_t) * max_target_q_values

        #Compute loss

        q_values = self.online_network(observations_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        #Loss, here we take the huber loss!

        loss = F.smooth_l1_loss(action_q_values, targets)
        
        #Uncomment the following code to use the MSE loss instead!
        #loss = F.mse_loss(action_q_values, targets)
        
        #Gradient descent to update the weights of the neural networ
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()
        
    def update_target_network(self):
        
        
        self.target_network.load_state_dict(self.online_network.state_dict())
    


    def train(
        self,
        policy,
        n_episodes: int,
        epsilon: float = 0.1,
        epsilon_decay: bool = False,
        alpha: float = 0.1,
        random_startpoint: bool = False,
        start_amount: float = 0.5,
        val_price_data:dict[datetime, float] | None = None,
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
            state = self.env.reset(
                random_startpoint=random_startpoint, start_amount=start_amount
            )

            if self.epsilon_decay:
                self.epsilon = epsilon_start * epsilon_decay_step**episode

            # play until episode is terminated
            terminated = False
            while not terminated:
                action = self.choose_action(state, policy)
                next_state, reward, terminated, *_ = self.env.step(action)
                #self.update_Q_table(state, action, reward, next_state)
                state = next_state

            # store episode data
            self.train_reward.append(self.env.episode_data.total_reward)

            # if (episode + 1) % 100 == 0:
            #    self.env.episode_data.plot()

            if val_env is not None:
                val_env.reset()
                terminated = False
                while not terminated:
                    action = self.choose_action(state, "greedy")
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
            action = self.choose_action(state, "greedy")
            next_state, _, terminated, *_ = self.env.step(action)
            state = next_state

        return self.env.episode_data  

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
        for q in self.env.quantiles:
            plt.axvline(q, color="red", lw=0.7)
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
        colors = {0: "white", 1: "green", 2: "red"}
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

      





def training_loop(self,  agent, max_episodes, target_ = False, seed=42):

    '''
    Params:
    env = name of the environment that the agent needs to play
    agent= which agent is used to train
    max_episodes = maximum number of games played
    target = boolean variable indicating if a target network is used (this will be clear later)
    seed = seed for random number generator for reproducibility
    
    Returns:
    average_reward_list = a list of averaged rewards over 100 episodes of playing the game
    '''
    env = self.env
    #env = gym.make(env_name, render_mode = None)
    env.action_space.seed(seed)
    obs = env.reset(seed=seed)
    average_reward_list = [-200]
    episode_reward = 0.0
    batch_size = 32
    dagent = agent

    
    for step in range(max_episodes):
        
        action= self.choose_action(step, obs)
    
        new_obs, rew, terminated,  _ = env.step(action)
        done = terminated       
        transition = (obs, action, rew, done, new_obs)
        self.replay_memory.add_data(transition)
        obs = new_obs
    
        episode_reward += rew
    
        if done:
        
            obs, _ = env.reset(seed=seed)
            self.replay_memory.add_reward(episode_reward)
            #Reinitilize the reward to 0.0 after the game is over
            episode_reward = 0.0

        #Learn

        self.learn(batch_size)

        #Calculate after each 100 episodes an average that will be added to the list
                
        if (step+1) % 100 == 0:
            average_reward_list.append(np.mean(self.replay_memory.reward_buffer))
        
        #Update target network, do not bother about it now!
        if target_:
            
            #Set the target_update_frequency
            target_update_frequency = 250
            if step % target_update_frequency == 0:
                self.update_target_network()
    
        #Print some output
        if (step+1) % 10000 == 0:
            print(20*'--')
            print('Step', step)
            #print('Epsilon', epsilon)
            print('Avg Rew', np.mean(self.replay_memory.reward_buffer))
            print()

    return average_reward_list