import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium import spaces

from src.utils import cumsum, joule_to_kwh


@dataclass
class DamEpisodeData:
    """Dataclass to store episode data for a dam environment"""

    storage: list[float] = field(default_factory=list)
    action: list[float] = field(default_factory=list)
    flow: list[float] = field(default_factory=list)
    price: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)

    def add(
        self, storage: float, action: float, flow: float, price: float, reward: float
    ):
        self.storage.append(storage)
        self.action.append(action)
        self.flow.append(flow)
        self.price.append(price)
        self.reward.append(reward)

    def plot(self):
        sns.set()
        fig, axs = plt.subplots(6, 1, figsize=(10, 10))

        axs[0].plot(self.storage)
        axs[0].set_title("Storage")

        axs[1].scatter(range(len(self.action)), self.action, s=1, marker="x")
        axs[1].set_title("Action")

        axs[2].plot(self.flow)
        axs[2].set_title("Flow")

        axs[3].plot(self.price)
        axs[3].set_title("Price")

        axs[4].plot(self.reward)
        axs[4].set_title("Reward")

        axs[5].plot(cumsum(self.reward))
        axs[5].set_title("Cumulative reward")

        fig.tight_layout()
        plt.show()


class DiscreteDamEnv(gym.Env):
    """Dam Environment that follows gym interface"""

    # static properties
    max_stored_energy = joule_to_kwh(100000 * 1000 * 9.81 * 30)  # U = mgh
    min_stored_energy = 0
    # a positive flow means emtpying the reservoir
    max_flow_rate = joule_to_kwh(5 * 3600 * 9.81 * 30)  # 5 m^3/s to m^3/h * gh

    buy_multiplier = 1.2  # i.e. we spend 1.2 Kw to store 1 Kw (80% efficiency)
    sell_multiplier = 0.9  # i.e. we get 0.9 Kw for selling 1 Kw (90% efficiency)

    price_bin_size = 200
    n_bins_reservoir = 10

    def __init__(self, price_data: dict[datetime, float]):
        super().__init__()

        self.price_data = price_data

        self.reset()

        # 0 = do nothing
        # 1 = empty / sell
        # 2 = fill / buy
        self.action_space = spaces.Discrete(3)

        # state is (hour, electricity price (bins), stored energy)
        n_bins_price = int(max(self.price_data.values()) // self.price_bin_size)

        self.observation_space = spaces.MultiDiscrete([24, n_bins_price+1, self.n_bins_reservoir+1])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, half_or_empty: str = "half",
              random_startpoint: bool = False):
        super().reset(seed=seed)

        if half_or_empty == "half":
            self.stored_energy = (
                    self.max_stored_energy / 2
            )  # start with a half-full reservor
        elif half_or_empty == "empty":
            self.stored_energy = 0

        self.current_date = min(self.price_data.keys())
        self.current_price = self.price_data[self.current_date]

        self.terminated = False
        self.episode_data = DamEpisodeData()

        if random_startpoint:
            return self.pick_random_startpoint()

        return self._get_state()

    def pick_random_startpoint(self):
        """Pick a random state to start from"""

        # start points to choose from
        start_points = list(self.price_data.keys())
        start_points = start_points[:-24]  # remove last day
        start = random.choice(start_points)

        # set the time variables
        self.current_date = start
        self.current_price = self.price_data[self.current_date]
        self.stored_energy = 0  # start at 0 to force the agent to fill the reservoir

        return self._get_state()

    def step(self, action: int):
        # empty reservor / sell
        if action == 1:
            flow_rate = self.max_flow_rate

        # fill reservor / buy
        elif action == 2:
            flow_rate = -self.max_flow_rate

        # do nothing, i.e. action == 0 (default)
        else:
            flow_rate = 0.0

        # update the applied flow so we don't overflow or store less than 0
        applied_flow_rate = self._apply_constrained_flow_rate(flow_rate)
        reward = self._get_reward(applied_flow_rate)

        # move to the next hour
        self._set_next_state()

        # store episode data
        self.episode_data.add(
            self.stored_energy,
            action,
            applied_flow_rate,
            self.current_price,
            reward,
        )

        # observation, reward, terminated, info (Gym convention)
        return (
            self._get_state(),
            reward,
            self.terminated,
            {},
        )

    def _apply_constrained_flow_rate(self, flow_rate: float):
        # positive flow means emtpying the reservoir
        self.stored_energy -= flow_rate

        # change flow rate if we overflow
        if self.stored_energy > self.max_stored_energy:
            correction = self.stored_energy - self.max_stored_energy
            flow_rate += correction
            self.stored_energy = self.max_stored_energy

        # change flow rate if we store less than 0
        elif self.stored_energy < self.min_stored_energy:
            correction = self.min_stored_energy - self.stored_energy
            flow_rate -= correction
            self.stored_energy = self.min_stored_energy

        return flow_rate

    def _set_next_state(self):
        self.current_date += timedelta(hours=1)

        if self.current_date in self.price_data:
            self.current_price = self.price_data[self.current_date]

        else:
            self.terminated = True

    def _get_state(self):
        return (self.current_date.hour, self._get_price_bin(), self._get_reservoir_bin())

    def _get_price_bin(self):
        return int(self.current_price // self.price_bin_size)

    def _get_reservoir_bin(self):
        return int(self.stored_energy // (self.max_stored_energy / self.n_bins_reservoir))

    def _get_reward(self, flow: float):
        # positive flow = selling
        if flow > 0:
            return flow * self.sell_multiplier * self.current_price

        # negative flow = buying
        return flow * self.buy_multiplier * self.current_price
