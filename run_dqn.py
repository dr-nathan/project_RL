import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from src.environment import DiscreteContinuousDamEnv
from src.utils import convert_dataframe
import torch
from src.agent_dqn import DDQNAgent


if __name__ == "__main__":

    seed_value = 7
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_data_path = Path(__file__).parent / "data" / "train.xlsx"
    train_data = pd.read_excel(train_data_path)
    train_data = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data = convert_dataframe(val_data)

    # load environment
    environment = DiscreteContinuousDamEnv(train_data)  # continuous states with discrete actions

    # load the DQN agent
    discount_rate = 0.98
    batch_size = 64

    epsilon = 0.5  # overwritten if epsilon_decay is True
    epsilon_start = 1
    epsilon_end = 0.05
    epsilon_decay = True

    lr = 5e-3
    n_episodes = int(20 * len(environment))  # number is how many times you run throuh the whole dataset
    buffer_size = len(environment)

    dagent = DDQNAgent(
        env=environment,
        device=device,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        n_episodes=n_episodes,
        discount_rate=discount_rate,
        lr=lr,
        buffer_size=buffer_size,
        seed=seed_value
    )

    episode_data = dagent.training_loop(batch_size, price_data_val=val_data)

    episode_data.debug_plot("Final training episode")
    
    episode_data = dagent.validate(price_data=val_data)

    episode_data.debug_plot("Validation episode")
    print(f"total val reward: {episode_data.total_reward}")
 