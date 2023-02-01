import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.agent.dqn import DDQNAgent
from src.environment.dam import DiscreteContinuousDamEnv
from src.utils import convert_dataframe

if __name__ == "__main__":

    seed_value = 7
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_data_path = Path(__file__).parent / "data" / "train.xlsx"
    train_data = pd.read_excel(train_data_path)
    train_data = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data = convert_dataframe(val_data)

    # load environment
    environment = DiscreteContinuousDamEnv(
        train_data
    )  # continuous states with discrete actions
    val_environment = DiscreteContinuousDamEnv(val_data)

    # load the DQN agent
    discount_rate = 0.98
    batch_size = 64

    epsilon = 0.5  # overwritten if epsilon_decay is True
    epsilon_start = 1
    epsilon_end = 0.05
    epsilon_decay = True

    lr = 5e-3
    n_episodes = int(
        20 * len(environment)
    )  # number is how many times you run throuh the whole dataset
    buffer_size = len(environment)

    agent = DDQNAgent(
        env=environment,
        val_env=val_environment,
        device=device,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        n_episodes=n_episodes,
        discount_rate=discount_rate,
        lr=lr,
        buffer_size=buffer_size,
        seed=seed_value,
    )

    # if file exists, load policy
    filepath = Path(__file__).parent / "models" / "DQN" / "model.pt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        agent.load(filepath)
        print("Loaded agent from file")

    train_filepath = filepath.parent / "training.pt"
    agent.training_loop(batch_size, save_path=train_filepath)
    agent.env.episode_data.debug_plot("Final training episode")

    agent.load(train_filepath)
    agent.validate(env=val_environment)
    val_environment.episode_data.debug_plot("Final validation episode")

    print(f"total val reward: {val_environment.episode_data.total_reward}")
