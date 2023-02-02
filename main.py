import argparse
from pathlib import Path

import torch

from src.agent.dqn import DDQNAgent
from src.environment.TestEnv import HydroElectric_Test
from src.environment.dam import TestEnvWrapper

data_path = Path(__file__).parent

parser = argparse.ArgumentParser()
# Path to the excel file with the train data
parser.add_argument(
    "--train_file",
    type=str,
    default=data_path / "data" / "train.xlsx",
    help="Path to the excel file with the train data",
)
# Path to the excel file with the validation
parser.add_argument(
    "--test_file",
    type=str,
    default=data_path / "data" / "validate.xlsx",
    help="Path to the excel file with the test data",
)
# If false, the model will only be validated
parser.add_argument(
    "--train", action="store_true", help="If false, the model will only be validated"
)
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.train_file)
val_env = HydroElectric_Test(path_to_test_data=args.test_file)

# NOTE: We wrap the provided TestEnv to make it compatible with the agent
env_wrapped = TestEnvWrapper(env)
val_env_wrapped = TestEnvWrapper(val_env)

# load the DQN agent
discount_rate = 0.99
batch_size = 64
epsilon = 0.5  # overwritten if epsilon_decay is True
epsilon_start = 1
epsilon_end = 0.05
epsilon_decay = True
lr = 5e-4
# number is how many times you run throuh the whole dataset
n_episodes = int(50 * len(env_wrapped))
buffer_size = len(env_wrapped)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 7

agent = DDQNAgent(
    env=env_wrapped,
    val_env=val_env_wrapped,
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

# if file exists, load policy
filepath = Path(__file__).parent / "models" / "test" / "dqn.pt"
filepath.parent.mkdir(parents=True, exist_ok=True)

if filepath.exists() and not args.train:
    agent.load(filepath)
    print("Loaded agent from file")
else:
    print("No agent found, creating new one")

if args.train:
    agent.training_loop(batch_size, save_path=filepath.parent / "training.pt")

total_reward, episode_data = agent.validate(val_env_wrapped)

episode_data.debug_plot()
print(f"Total reward: {total_reward}")
