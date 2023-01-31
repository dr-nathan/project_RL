from src.TestEnv import HydroElectric_Test
import argparse
import torch
from src.agent_dqn import DDQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='data/train.xlsx')  # Path to the excel file with the train data
parser.add_argument('--val_file', type=str, default='data/validate.xlsx')  # Path to the excel file with the validation
parser.add_argument('--train', type=bool, default=False)  # If false, the model will only be validated
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.train_file)
val_env = HydroElectric_Test(path_to_test_data=args.val_file)

# load the DQN agent
discount_rate = 0.995
batch_size = 32
epsilon = 0.5  # overwritten if epsilon_decay is True
epsilon_start = 1
epsilon_end = 0.05
epsilon_decay = True
lr = 5e-4
env.len = env.price_values.shape[0] * env.price_values.shape[1]
n_episodes = int(20 * env.len)  # number is how many times you run throuh the whole dataset
buffer_size = env.len
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_value = 7

dagent = DDQNAgent(
    env=env,
    val_env=val_env,
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

if not args.train:
    dagent.training_loop(batch_size)

total_reward, episode_data = dagent.validate(val_env)

episode_data.debug_plot()
print(f'Total reward: {total_reward}')
