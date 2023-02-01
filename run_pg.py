from pathlib import Path

import pandas as pd

from src.agent.pg import BasicPGAgent
from src.environment.dam import ContinuousDamEnv
from src.utils import convert_dataframe

if __name__ == "__main__":
    # load data
    train_data = pd.read_excel(Path(__file__).parent / "data" / "train.xlsx")
    train_data = convert_dataframe(train_data)

    val_data = pd.read_excel(Path(__file__).parent / "data" / "validate.xlsx")
    val_data = convert_dataframe(val_data)

    # create environment and agent
    environment = ContinuousDamEnv(train_data)
    agent = BasicPGAgent(
        env=environment,
        discount_factor=0.98,
        epochs=5,
        hidden_layers=3,
        hidden_size=5,
        learning_rate=1e-3,
    )

    # train agent
    n_episodes = 1

    # if file exists, load policy
    filepath = Path(__file__).parent / "models" / "PG" / "model.pt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        agent.load(filepath)
        print("Loaded agent from file")

    # train
    agent.train(n_episodes, filepath.parent)
    agent.save(filepath)
    agent.env.episode_data.debug_plot("Final training episode")

    # validate
    agent.validate(price_data=val_data)
    agent.env.episode_data.debug_plot("Validation episode")
