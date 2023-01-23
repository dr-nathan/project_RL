import pandas as pd
from pathlib import Path
#from src.agent import QLearnAgent
from src.environment import ContinuousDamEnv, DiscreteContinuousDamEnv, DiscreteDamEnv
from src.utils import convert_dataframe
import torch
from src.Agent_dqn import DDQNAgent


if __name__ == "__main__": 
    #Set the hyperparameters

    #Discount rate
    discount_rate = 0.99
    #That is the sample that we consider to update our algorithm
    batch_size = 32
    #Maximum number of transitions that we store in the buffer
    buffer_size = 50000
    #Minimum number of random transitions stored in the replay buffer
    min_replay_size = 1000
    #Starting value of epsilon
    epsilon_start = 1.0
    #End value (lowest value) of epsilon
    epsilon_end = 0.05
    #Decay period until epsilon start -> epsilon end
    epsilon_decay = 10000

    max_episodes = 250000 
    lr = 5e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_data_path = Path(__file__).parent / "data" / "train.xlsx"
    train_data = pd.read_excel(train_data_path)
    train_data = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data = convert_dataframe(val_data)

    #load environment 

    environment = DiscreteContinuousDamEnv(train_data)

    #load QL Agent - get the training function
    #agent = QLearnAgent(environment)

    #load the DQN agent
    dagent = DDQNAgent(env_name= environment, device=device, epsilon_decay=epsilon_decay, epsilon_start=epsilon_start, epsilon_end=epsilon_end, discount_rate=discount_rate, lr=lr, buffer_size=buffer_size)
    #average_rewards_ddqn = training_loop(environment, dagent, max_episodes, target_ = True) 


        # train agent
    epsilon_decay = False
    epsilon = 0.8  # overriden if epsilon_decay is True
    alpha = 0.1
    n_episodes = 10
    random_startpoint = False
    start_amount = 0.5

    
    dagent.train(
        "epsilon_greedy",
        n_episodes,
        epsilon,
        epsilon_decay,
        alpha,
        random_startpoint,
        start_amount,
        val_price_data=val_data
    )
    #episode_data = agent.train("epsilon_greedy", 30, epsilon=0.2, alpha=0.1)
    
    #dagent.training_loop(environment,  max_episodes=n_episodes, target_ = True)
    
    print(f"Total reward: {dagent.env.episode_data.total_reward}")
    #agent.env.episode_data.plot()
    dagent.env.episode_data.debug_plot()
 