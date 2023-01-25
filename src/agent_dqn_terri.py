from tqdm import tqdm
from datetime import datetime
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class DQN(nn.Module):
    
    def __init__(self, env, learning_rate):

        super().__init__()
        input_features, *_ = env.observation_space.shape
        action_space = env.action_space.n
        
        self.dense1 = nn.Linear(in_features=input_features, out_features=32)
        # self.dense2 = nn.Linear(in_features=128, out_features=64)
        #self.dense3 = nn.Linear(in_features=64, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=action_space)
        
        # Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):

        x = torch.relu(self.dense1(x))
        # x = torch.relu(self.dense2(x))
        #x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        
        return x


""" class ExperienceReplay:
    
    def __init__(self, env, replay_size = 1000):

        


        self.env = env
        self.replay_size = replay_size
        self.episode_rewards = []

        self.reset()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fill_replay_memory(self):


        env = self.env
        state = env.reset( )  # TODO: implement random startpoint
        episode_reward = 0.0

        for i in range(self.replay_size):

            action = self.env.action_space.sample()
            next_state, reward, terminated, _ = env.step(action)
            transition = (state, action, reward, terminated, next_state)
            self.replay_memory.append(transition)
            state = next_state

            episode_reward += reward

            if terminated:
                state = env.reset()
                self.episode_rewards.append(episode_reward)
                episode_reward = 0.0

    def reset(self):
 
        self.replay_memory = []
        self.episode_rewards = []
            
    def sample(self, batch_size):

        # sample random transitions from the replay memory
        transitions = random.sample(self.replay_memory, batch_size)

        # Solution
        observations = [t[0] for t in transitions]
        actions = [t[1] for t in transitions]
        rewards = [t[2] for t in transitions]
        dones = [t[3] for t in transitions]
        new_observations = [t[4] for t in transitions]

        # PyTorch needs these arrays as tensors!, don't forget to specify the device! (cpu / GPU)
        observations_t = torch.as_tensor(observations, dtype = torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype = torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype = torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype = torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype = torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, dones_t, new_observations_t """

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
        self.seed = seed
        
    def fill_replay_memory(self):
                
        obs = self.env.reset(seed=self.seed)
        for _ in range(self.min_replay_size):
            
            action = self.env.action_space.sample()
            new_obs, rew, terminated,_ = self.env.step(action)
            done = terminated 

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
    
            if done:
                obs = self.env.reset(seed=self.seed)

    def get_obs(self):
        return self.env.reset(seed=self.seed)


        

      
          
    def add_data(self, data): 
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        self.replay_buffer.append(data)

    def reset(self):

        self.replay_buffer = []
        self.episode_rewards = []    
            
    def sample(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        
        Returns:
        tensor of observations, actions, rewards, done (boolean) and next observation 
        '''
        
        transitions = random.sample(self.replay_buffer, batch_size)

        #Solution
        observations = [t[0] for t in transitions]
        actions = [t[1] for t in transitions]
        rewards = [t[2] for t in transitions]
        dones = [t[3] for t in transitions]
        new_observations = [t[4] for t in transitions]

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
    
    def __init__(self, env, device, epsilon, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size):
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
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        
        self.replay_buffer = ExperienceReplay(self.env, self.buffer_size)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)

        self.target_network = DQN(self.env, self.learning_rate).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def training_loop(self, iterations):

        for iteration in tqdm(range(iterations)):

            self.replay_buffer.reset()
            #self.replay_buffer.fill_replay_memory()
            self.fill_action()

            self.learn(batch_size=64)

            if (iteration+1) % 10 == 0:

                self.update_target_network()



        """     def training_loop(self, iterations):
        

        for step in  tqdm(range(iterations)):
            self.replay_buffer.reset()
            obs = self.replay_buffer.get_obs()

            action = self.choose_action(obs, "epsilon_greedy")
            new_obs, rew, terminated,_ = self.env.step(action)
            done = terminated 

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.add_data(transition)
            obs = new_obs
    
            if done:
                obs = self.replay_buffer.get_obs()

            self.learn(batch_size=32)
            if (step+1) % 250 == 0:
                self.update_target_network() """

    def fill_action(self):
                        
        obs = self.replay_buffer.get_obs()
        for _ in range(self.replay_buffer.min_replay_size):
            
            action = self.choose_action(obs, "greedy")
            new_obs, rew, terminated,_ = self.env.step(action)
            done = terminated 

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.add_data(transition)
            obs = new_obs
    
            if done:
                obs = self.replay_buffer.get_obs()


       

    def learn(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        '''
        
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_buffer.sample(batch_size)

        #Compute targets, note that we use the same neural network to do both! This will be changed later!

        target_q_values = self.target_network(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1-dones_t) * max_target_q_values

        #Compute loss

        q_values = self.online_network(observations_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        #Loss, here we take the huber loss!

        #loss = F.smooth_l1_loss(action_q_values, targets)
        
        #Uncomment the following code to use the MSE loss instead!
        loss = F.mse_loss(action_q_values, targets)
        
        #Gradient descent to update the weights of the neural networ
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
            return self.online_network(state).argmax().item()
        elif policy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                return self.online_network(state).argmax().item()

        else:
            raise ValueError("Unknown policy")

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
        print(self.env.episode_data.rewards)
        return self.env.episode_data  

# TODO: has to be updated to work with new agent
    def plot_rewards_over_episode(self):
        plt.plot(self.train_reward, label="Train")
        plt.plot(self.val_reward, label="Validation")
        plt.legend()
        plt.title("Total reward over episode")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.show()


