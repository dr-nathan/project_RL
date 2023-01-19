from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.agent import QLearnAgent
from src.environment import DiscreteDamEnv
from src.utils import convert_dataframe

if __name__ == "__main__":

    DEBUG = True

    # load data
    train_data = pd.read_excel(Path(__file__).parent / "data" / "train.xlsx")
    train_data = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data = convert_dataframe(val_data)

    # create environment and agent
    environment = DiscreteDamEnv(train_data)
    agent = QLearnAgent(environment, 0.98)

    # train agent
    epsilon_decay = False
    epsilon = 0.6  # overriden if epsilon_decay is True
    alpha = 0.1
    n_episodes = 300
    random_startpoint = False
    start_amount = 0.5

    agent.train(
        "epsilon_greedy",
        n_episodes,
        epsilon,
        epsilon_decay,
        alpha,
        random_startpoint,
        start_amount,
        val_price_data=val_data
    )

    if DEBUG:
        agent.env.episode_data.debug_plot("Final training episode")
        agent.plot_rewards_over_episode()
        # agent.env.plot_price_distribution() # only actually relevant for baseline insight

    # validate agent
    agent.validate(price_data=val_data)
    if DEBUG:
        agent.env.episode_data.debug_plot("Validation episode")
        agent.plot_price_bins(train_data, val_data)

    # print total reward
    print(f"Total reward: {agent.env.episode_data.total_reward}")

    # plot Q table
    agent.visualize_Q_table()
    agent.env.episode_data.plot_fancy()
