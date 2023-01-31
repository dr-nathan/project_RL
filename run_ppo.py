from pathlib import Path

import pandas as pd

from src.agent_pg import PPOAgent
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
    agent = PPOAgent(
        env=environment,
        clip_epsilon=0.2,
        discount_factor=0.98,
        entropy_loss_coeff=0.01,
        epochs=5,
        hidden_layers=3,
        hidden_size=5,
        learning_rate=1e-3,
    )

    # train agent
    n_episodes = 500
    random_startpoint = False

    # if file exists, load policy
    filepath = Path(__file__).parent / "PPO" / "model.pt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        agent.load(filepath)
        print("Loaded agent from file")

    # train agent
    agent.train(n_episodes, save_path=filepath.parent)
    agent.save(filepath)
    agent.env.episode_data.debug_plot("Final training episode")

    # validate agent
    agent.validate(price_data=val_data)
    agent.env.episode_data.debug_plot("Validation episode")
