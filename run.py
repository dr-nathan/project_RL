from pathlib import Path

import pandas as pd

from src.agent import Agent
from src.environment import DiscreteDamEnv
from src.utils import convert_dataframe

if __name__ == "__main__":

    # load data
    train_data = pd.read_excel(Path(__file__).parent / "data" / "train.xlsx")
    train_data, train_data_real = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data, val_data_real = convert_dataframe(val_data)

    # create environment and agent
    environment = DiscreteDamEnv(train_data, train_data_real)
    agent = Agent(environment)

    # train agent
    epsilon_decay = True
    epsilon = 0.2  # overriden if epsilon_decay is True
    alpha = 0.3
    n_episodes = 100
    random_startpoint = False

    episode_data = agent.train(
        "epsilon_greedy", n_episodes, epsilon, epsilon_decay, alpha, random_startpoint
    )


    agent.env.episode_data.plot()
    agent.env.plot_price_distribution()
    agent.env.episode_data.plot("Final training episode")

    # validate agent
    # re-make env with validation data
    environment = DiscreteDamEnv(val_data, val_data_real)
    agent = Agent(environment)
    agent.validate()
    agent.env.episode_data.plot("Validation episode")

