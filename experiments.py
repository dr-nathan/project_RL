import copy
import os

import matplotlib.pyplot as plt
import torch

from src.agent.dqn import DDQNAgent
from src.agent.dqn_small import DDQNAgentSmall
from src.environment.dam import TestEnvWrapper
from src.environment.dam_small import TestEnvWrapperSmall
from src.environment.TestEnv import HydroElectric_Test

env = HydroElectric_Test(path_to_test_data="data/train.xlsx")
val_env = HydroElectric_Test(path_to_test_data="data/validate.xlsx")

env_wrapped = TestEnvWrapper(copy.deepcopy(env))
val_env_wrapped = TestEnvWrapper(copy.deepcopy(val_env))
env_wrapped_small = TestEnvWrapperSmall(copy.deepcopy(env))
val_env_wrapped_small = TestEnvWrapperSmall(copy.deepcopy(val_env))

# load the DQN agent
discount_rate = 0.99
batch_size = 64
epsilon = 0.5  # overwritten if epsilon_decay is True
epsilon_start = 1
epsilon_end = 0.05
epsilon_decay = True
lr = 5e-4
# number is how many times you run throuh the whole dataset
n_episodes = int(10 * len(env_wrapped))
buffer_size = len(env_wrapped)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 7
#
# # train the agent 10 times
# for i in range(10):
#     agent = DDQNAgent(
#         env=copy.deepcopy(env_wrapped),
#         val_env=copy.deepcopy(val_env_wrapped),
#         device=device,
#         epsilon=epsilon,
#         epsilon_decay=epsilon_decay,
#         epsilon_start=epsilon_start,
#         epsilon_end=epsilon_end,
#         n_episodes=n_episodes,
#         discount_rate=discount_rate,
#         lr=lr,
#         buffer_size=buffer_size,
#         seed=seed_value,
#     )
#     train_filepath = f"models/experiment/small_small{i}.pt"
#     agent.training_loop(batch_size, save_path=train_filepath)

# make boxplot of the results
# for every stored model, run the validation loop and store the results
rewards_big_big = []
rewards_big_small = []
rewards_small_big = []
rewards_small_small = []

agent = DDQNAgentSmall(
    env=copy.deepcopy(env_wrapped_small),
    val_env=copy.deepcopy(val_env_wrapped_small),
    device=device,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    n_episodes=n_episodes,
    discount_rate=discount_rate,
    lr=lr,
    buffer_size=buffer_size,
    seed=seed_value,
)
for model in os.listdir("models/experiment"):
    if "small_small" in model:
        agent.load(f"models/experiment/{model}")
        total_reward, episode_data = agent.validate(val_env_wrapped_small)
        rewards_small_small.append(total_reward)


agent = DDQNAgentSmall(
    env=copy.deepcopy(env_wrapped),
    val_env=copy.deepcopy(val_env_wrapped),
    device=device,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    n_episodes=n_episodes,
    discount_rate=discount_rate,
    lr=lr,
    buffer_size=buffer_size,
    seed=seed_value,
)
for model in os.listdir("models/experiment"):
    if "small_large" in model:
        agent.load(f"models/experiment/{model}")
        total_reward, episode_data = agent.validate(val_env_wrapped)
        rewards_small_big.append(total_reward)

agent = DDQNAgent(
    env=copy.deepcopy(env_wrapped_small),
    val_env=copy.deepcopy(val_env_wrapped_small),
    device=device,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    n_episodes=n_episodes,
    discount_rate=discount_rate,
    lr=lr,
    buffer_size=buffer_size,
    seed=seed_value,
)
for model in os.listdir("models/experiment"):
    if "large_small" in model:
        agent.load(f"models/experiment/{model}")
        total_reward, episode_data = agent.validate(val_env_wrapped_small)
        rewards_big_small.append(total_reward)

agent = DDQNAgent(
    env=copy.deepcopy(env_wrapped),
    val_env=copy.deepcopy(val_env_wrapped),
    device=device,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    n_episodes=n_episodes,
    discount_rate=discount_rate,
    lr=lr,
    buffer_size=buffer_size,
    seed=seed_value,
)
for model in os.listdir("models/experiment"):
    if "large_large" in model:
        agent.load(f"models/experiment/{model}")
        total_reward, episode_data = agent.validate(val_env_wrapped)
        rewards_big_big.append(total_reward)

# make boxplot of the results
plt.boxplot(
    [rewards_big_big, rewards_big_small, rewards_small_big, rewards_small_small]
)
# name the models, set name vertically
plt.xticks([1, 2, 3, 4], ["NN+ FS+", "NN+ FS-", "NN- FS+", "NN- FS-"])
plt.title("Boxplot of the rewards of the different models")
plt.ylabel("Reward")
plt.xlabel("Model")
plt.show()
