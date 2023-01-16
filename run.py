from src.agent import Agent
from src.environment import DiscreteDamEnv
from src.utils import convert_dataframe

import pandas as pd
from pathlib import Path

if __name__ == "__main__":  

    # load data
    train_data_path = Path(__file__).parent / "data" / "train.xlsx"
    train_data = pd.read_excel(train_data_path)
    train_data = convert_dataframe(train_data)

    environment = DiscreteDamEnv(train_data)

    agent = Agent(environment)
    episode_data = agent.train("epsilon_greedy", 500, epsilon=0.2, alpha=0.2)

    print(agent.Qtable)
    agent.env.episode_data.plot()
