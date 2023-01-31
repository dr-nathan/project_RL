# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.agent_pg import DEVICE, BasicPGNetwork

filepath = Path(__file__).parent / "PG" / "nobatch_lmean.pt"
net = BasicPGNetwork(2, 1, 5, 2).to(DEVICE)
net.load_state_dict(torch.load(filepath, map_location=DEVICE))

# %%
hours = torch.linspace(-10, 10, 1000)
prices = torch.linspace(-10, 10, 1000)

x, y = torch.meshgrid(hours, prices, indexing="xy")

inp = torch.stack([x, y], dim=2).to(DEVICE)
means, stds = net(inp)

# %%
fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(
    x, y, means.squeeze(-1).cpu().detach().numpy(), cmap="viridis", edgecolor="none"
)
ax.set_title("Mean")
ax.set_xlabel("Hours")
ax.set_ylabel("Prices")
ax.set_zlabel("$\mu$")
ax.dist = 11

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.plot_surface(
    x, y, stds.squeeze(-1).cpu().detach().numpy(), cmap="viridis", edgecolor="none"
)
ax.set_title("Std")
ax.set_xlabel("Hours")
ax.set_ylabel("Prices")
ax.set_zlabel("$\sigma$")
ax.dist = 11

fig.tight_layout()
fig.savefig("pg_output.pdf")
fig.savefig("pg_output.png")

# %%
