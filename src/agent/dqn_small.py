import torch
from torch import nn, optim

from src.agent.dqn import DDQNAgent


class DQNSmall(nn.Module):
    def __init__(self, env, learning_rate, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # NV: get the input features selected by agent
        input_features, *_ = env.observation_space.shape
        action_space = env.action_space.n

        self.dense1 = nn.Linear(in_features=input_features, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = self.dense4(x)

        return x


class DDQNAgentSmall(DDQNAgent):
    def __init__(
        self,
        env,
        val_env,
        device: torch.device,
        epsilon: float = 1.0,
        epsilon_decay: bool = True,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        n_episodes: int = 20,
        discount_rate: float = 0.99,
        lr: float = 5e-4,
        buffer_size: int = 100000,
        seed: int = 7,
    ):
        super().__init__(
            env,
            val_env,
            device,
            epsilon,
            epsilon_decay,
            epsilon_start,
            epsilon_end,
            n_episodes,
            discount_rate,
            lr,
            buffer_size,
            seed,
        )

        self.online_network = DQNSmall(self.env, self.learning_rate, seed=self.seed).to(
            self.device
        )

        self.target_network = DQNSmall(self.env, self.learning_rate, seed=self.seed).to(
            self.device
        )
        self.target_network.load_state_dict(self.online_network.state_dict())
