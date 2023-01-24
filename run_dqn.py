import pandas as pd
from pathlib import Path
from src.environment import ContinuousDamEnv, DiscreteContinuousDamEnv, DiscreteDamEnv
from src.utils import convert_dataframe
import torch
from src.agent_dqn import DDQNAgent


if __name__ == "__main__":

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
    epsilon = 0.1
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 10000
    lr = 5e-4

    dagent = DDQNAgent(
        env=environment,
        device=device,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        discount_rate=discount_rate,
        lr=lr,
        buffer_size=len(train_data)
    )

    n_episodes = 1000

    dagent.training_loop(n_episodes)
    
    episode_data = dagent.validate(price_data=val_data)

    episode_data.debug_plot("Validation episode")
 