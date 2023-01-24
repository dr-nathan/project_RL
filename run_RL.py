import pandas as pd
from pathlib import Path
from src.environment import ContinuousDamEnv
from src.utils import convert_dataframe
import torch
from src.agent_continuous import DDQNAgent, training_loop


if __name__ == "__main__": 

    # Discount rate
    discount_rate = 0.99
    # That is the sample that we consider to update our algorithm
    batch_size = 32
    # Maximum number of transitions that we store in the buffer
    buffer_size = 50000
    # Minimum number of random transitions stored in the replay buffer
    min_replay_size = 1000
    # Starting value of epsilon
    epsilon_start = 1.0
    # End value (lowest value) of epsilon
    epsilon_end = 0.05
    # Decay period until epsilon start -> epsilon end
    epsilon_decay = 10000

    max_episodes = 250000 
    lr = 5e-4

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # load data
    train_data_path = Path(__file__).parent / "data" / "train.xlsx"
    train_data = pd.read_excel(train_data_path)
    train_data = convert_dataframe(train_data)

    environment = ContinuousDamEnv(train_data)

    dagent = DDQNAgent(
        env_name=environment,
        device=device,
        epsilon_decay=epsilon_decay,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        discount_rate=discount_rate,
        lr=lr,
        buffer_size=buffer_size
    )
    average_rewards_ddqn = training_loop(environment, dagent, max_episodes, target_=True)
    # episode_data = agent.train("epsilon_greedy", 30, epsilon=0.2, alpha=0.1)
  
    #agent.env.episode_data.plot()
    dagent.env.episode_data.plot()