from pathlib import Path

import pandas as pd

from src.agent import PolicyGradientAgent
from src.environment import DiscreteDamEnv
from src.utils import convert_dataframe

if __name__ == "__main__":

    # load data
    train_data = pd.read_excel(Path(__file__).parent / "data" / "train.xlsx")
    train_data = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data = convert_dataframe(val_data)

    # determine quantile to cap the price for the bins at
    # price_quantile = pd.Series(train_data.values()).quantile(0.99)

    # create environment and agent
    environment = DiscreteDamEnv(train_data, 200)
    agent = PolicyGradientAgent(learning_rate=0.1, env=environment)

    # train agent
    epsilon_decay = True
    epsilon = 0.2  # overriden if epsilon_decay is True
    alpha = 0.3
    n_episodes = 20
    random_startpoint = False

    episode_data = agent.train(n_episodes)

    agent.env.plot_price_distribution()
    agent.env.episode_data.plot_debug("Final training episode")

    # validate agent
    agent.validate(price_data=val_data)
    agent.env.episode_data.plot_debug("Validation episode")
