# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

from src.agent.pg import DEVICE


class BasicPGNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.ReLU()

        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_std = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        mean = self.fc_mean(x)
        std = self.fc_std(x)
        std = torch.exp(std)
        std = torch.clamp(std, min=1e-8)  # std cannot be < 0
        return mean, std


filepath = Path(__file__).parent / "models" / "PG" / "nobatch_lmean.pt"
net = BasicPGNetwork(2, 1, 5).to(DEVICE)
net.load_state_dict(torch.load(filepath, map_location=DEVICE))

# %%
hours = torch.linspace(-10, 10, 1000)
prices = torch.linspace(-10, 10, 1000)

hsmall = torch.linspace(0, 1, 1000)
psmall = torch.linspace(0, 1, 1000)

x, y = torch.meshgrid(hours, prices, indexing="xy")
xs, ys = torch.meshgrid(hsmall, psmall, indexing="xy")

inp = torch.stack([x, y], dim=2).to(DEVICE)
means, stds = net(inp)

inps = torch.stack([xs, ys], dim=2).to(DEVICE)
meanss, stdss = net(inps)

# %%
sns.set(style="whitegrid")
fig = plt.figure(figsize=(8.8, 4.8))

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(
    x, y, means.squeeze(-1).cpu().detach().numpy(), cmap="viridis", edgecolor="none"
)
ax.set_title("Policy output for enlarged state space")
ax.set_xlabel("Hours")
ax.set_ylabel("Prices")
ax.set_zlabel("$\mu$")
ax.dist = 11

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.plot_surface(
    xs, ys, meanss.squeeze(-1).cpu().detach().numpy(), cmap="viridis", edgecolor="none"
)
ax.set_title("Policy output for actual state space")
ax.set_xlabel("Hours")
ax.set_ylabel("Prices")
ax.set_zlabel("$\mu$")
ax.dist = 11

fig.tight_layout()
fig.savefig("pg_output.pdf")
fig.savefig("pg_output.png")

# %%
