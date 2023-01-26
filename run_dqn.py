import pandas as pd
from pathlib import Path
from src.environment import DiscreteContinuousDamEnv
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
    batch_size = 32

    epsilon = 0.5  # overwritten if epsilon_decay is True
    epsilon_start = 1
    epsilon_end = 0.05
    epsilon_decay = True

    lr = 5e-3
    n_episodes = 100000
    buffer_size = 25000

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
        buffer_size=buffer_size
    )

    episode_data = dagent.training_loop(batch_size)

    episode_data.debug_plot("Final training episode")
    
    episode_data = dagent.validate(price_data=val_data)

    episode_data.debug_plot("Validation episode")
 