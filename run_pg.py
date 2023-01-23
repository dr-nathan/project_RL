from pathlib import Path

import pandas as pd

from src.agent_continuous import PolicyGradientAgent
from src.environment import ContinuousDamEnv
from src.utils import convert_dataframe

if __name__ == "__main__":

    # load data
    train_data = pd.read_excel(Path(__file__).parent / "data" / "train.xlsx")
    train_data = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data = convert_dataframe(val_data)

    # create environment and agent
    environment = ContinuousDamEnv(train_data)
    agent = PolicyGradientAgent(learning_rate=0.1, env=environment)

    # train agent
    n_episodes = 10
    random_startpoint = False

    episode_data = agent.train(n_episodes)

    # agent.env.plot_price_distribution()
    agent.env.episode_data.debug_plot("Final training episode")

    # validate agent
    agent.validate(price_data=val_data)
    agent.env.episode_data.debug_plot("Validation episode")
