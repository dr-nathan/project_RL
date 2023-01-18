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

    # determine quantile to cap the price for the bins at
    # price_quantile = pd.Series(train_data.values()).quantile(0.99)

    # create environment and agent
    environment = DiscreteDamEnv(train_data, 200)
    agent = QLearnAgent(environment)

    # train agent
    epsilon_decay = False
    epsilon = 1  # overriden if epsilon_decay is True
    alpha = 0.1
    n_episodes = 700
    random_startpoint = False

    agent.train(
        "epsilon_greedy", n_episodes, epsilon, epsilon_decay, alpha, random_startpoint, val_price_data=val_data
    )

    if DEBUG:
        agent.plot_rewards_over_episode()
        agent.env.plot_price_distribution()
        agent.env.episode_data.debug_plot("Final training episode")

    # validate agent
    agent.validate(price_data=val_data)
    agent.env.episode_data.debug_plot("Validation episode")

    # plot Q table
    agent.visualize_Q_table()

