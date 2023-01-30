from pathlib import Path

import pandas as pd

from src.agent_pg import BasicPGAgent
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
    agent = BasicPGAgent(environment)

    # train agent
    n_episodes = 500

    # if file exists, agent
    filepath = Path(__file__).parent / "PG" / "model.pt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        agent.load(filepath)
        print("Loaded agent from file")

    # train
    if True:
        agent.train(n_episodes, filepath.parent)
        agent.save(filepath)
        agent.env.episode_data.debug_plot("Final training episode")

    # validate
    if True:
        agent.validate(price_data=val_data)
        agent.env.episode_data.debug_plot("Validation episode")
