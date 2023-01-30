from src.TestEnv import HydroElectric_Test
import argparse
import torch
from src.agent_dqn import DDQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='data/train.xlsx')  # Path to the excel file with the train data
parser.add_argument('--val_file', type=str, default='data/validate.xlsx')  # Path to the excel file with the validation
parser.add_argument('--validation', type=bool, default=False)  # If true, the model will be validated on the validation
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.train_file)

# load the DQN agent
discount_rate = 0.98
batch_size = 32
epsilon = 0.5  # overwritten if epsilon_decay is True
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = True
lr = 5e-3
n_episodes = int(10 * len(env))  # number is how many times you run throuh the whole dataset
buffer_size = len(env)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_value = 7

dagent = DDQNAgent(
    env=env,
    device=device,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    n_episodes=n_episodes,
    discount_rate=discount_rate,
    lr=lr,
    buffer_size=buffer_size,
    seed=seed_value
)

if not args.validation:
    episode_data = dagent.training_loop(batch_size, price_data_val=args.val_file)

    episode_data.debug_plot("Final training episode")

episode_data = dagent.validate(price_data=args.val_file)

episode_data.debug_plot("Validation episode")
print(f"total val reward: {episode_data.total_reward}")





