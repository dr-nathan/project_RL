import pandas as pd
from pathlib import Path
from src.agent import Agent
from src.environment import DiscreteDamEnv
from src.utils import convert_dataframe
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # load data
    train_data = pd.read_excel(Path(__file__).parent / "data" / "train.xlsx")
    train_data = convert_dataframe(train_data)

    # create environment and agent
    environment = DiscreteDamEnv(train_data)
    agent = Agent(environment)

    # train agent
    epsilon_decay = True
    epsilon = 0.5  # overriden if epsilon_decay is True
    alpha = 0.3
    n_episodes = 100
    random_startpoint = True

    episode_data = agent.train(
        "epsilon_greedy",
        n_episodes,
        epsilon,
        epsilon_decay,
        alpha,
        random_startpoint,
    )

    agent.env.episode_data.plot()

    q = agent.Qtable
    q_mean = q.mean(
        axis=tuple(range(1, len(agent.env.observation_space.nvec)))
    )  # collapse all but the first and last dimension
    plt.plot(q_mean[:, 0], label="buy")
    plt.plot(q_mean[:, 1], label="sell")
    plt.ylabel("Q-value")
    plt.xlabel("Hour")
    plt.legend()
    plt.plot()
