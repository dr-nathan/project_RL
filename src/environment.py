import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium import spaces

from src.utils import cumsum, joule_to_mwh


@dataclass
class DamEpisodeData:
    """Dataclass to store episode data for a dam environment"""

    date: list[datetime] = field(default_factory=list)
    storage: list[float] = field(default_factory=list)
    action: list[float] = field(default_factory=list)
    flow: list[float] = field(default_factory=list)
    price: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)

    def add(
        self,
        date: datetime,
        storage: float,
        action: float,
        flow: float,
        price: float,
        reward: float,
    ):
        self.date.append(date)
        self.storage.append(storage)
        self.action.append(action)
        self.flow.append(flow)
        self.price.append(price)
        self.reward.append(reward)

    def plot(self, title: str | None = None):
        sns.set()
        fig, axs = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

        if title:
            fig.suptitle(title)

        axs[0].plot(self.date, self.storage)
        axs[0].set_title("Storage")

        axs[1].scatter(self.date, self.action, s=1, marker="x")
        axs[1].set_title("Action")

        axs[2].plot(self.date, self.flow)
        axs[2].set_title("Flow")

        axs[3].plot(self.date, self.price)
        axs[3].set_title("Price")

        axs[4].plot(self.date, self.reward)
        axs[4].set_title("Reward")

        axs[5].plot(self.date, cumsum(self.reward))
        axs[5].set_title("Cumulative reward")

        fig.tight_layout()
        plt.show()


class DiscreteDamEnv(gym.Env):
    """Dam Environment that follows gym interface"""

    # static properties
    max_stored_energy = joule_to_mwh(
        100000 * 1000 * 9.81 * 30
    )  # 100000 m^3 to mwh with U = mgh
    min_stored_energy = 0
    # a positive flow means emtpying the reservoir
    max_flow_rate = joule_to_mwh(5 * 3600 * 1000 * 9.81 * 30)  # 5 m^3/s to mwh

    buy_multiplier = 1.2  # i.e. we spend 1.2 Kw to store 1 Kw (80% efficiency)
    sell_multiplier = 0.9  # i.e. we get 0.9 Kw for selling 1 Kw (90% efficiency)

    price_bin_size = 200
    n_bins_reservoir = 10

    def __init__(self, price_data: dict[datetime, float]):
        super().__init__()

        self.reset(price_data=price_data)

        # 0 = do nothing
        # 1 = empty / sell
        # 2 = fill / buy
        self.action_space = spaces.Discrete(3)

        # state is (hour, electricity price (bins), stored energy)
        n_bins_price = int(max(self.price_data.values()) // self.price_bin_size)

        self.observation_space = spaces.MultiDiscrete(
            [24, 12, n_bins_price + 1, self.n_bins_reservoir + 1]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        start_amount: float | Literal["random"] = 0.5,
        random_startpoint: bool = False,
        price_data: dict[datetime, float] | None = None,
    ):
        super().reset(seed=seed)

        if price_data:
            self.price_data = dict(sorted(price_data.items()))
        else:
            assert self.price_data and len(self.price_data) > 0, "No price data provided"

        # reservor starting level
        if start_amount == "random":
            start_amount = random.uniform(0, 1)

        self.stored_energy = self.max_stored_energy * start_amount

        self.current_date = min(self.price_data.keys())
        self.current_price = self.price_data[self.current_date]

        self.terminated = False
        self.episode_data = DamEpisodeData()

        # dealing with the iterator is a bit cumbersome, but much faster
        self._set_state_iter(self.price_data)

        if random_startpoint:
            return self.pick_random_startpoint()

        return self._get_state()

    def pick_random_startpoint(self):
        """Pick a random state to start from"""

        # start points to choose from
        start_points = list(self.price_data.keys())
        start_points = start_points[:-24]  # remove last day
        start_date = start_points[random.randint(0, len(start_points) - 1)]

        price_data = {k: v for k, v in self.price_data.items() if k >= start_date}
        self._set_state_iter(price_data)

        # set the time variables
        self.current_date = start_date
        self.current_price = self.price_data[self.current_date]

        # set the reservoir level
        self.stored_energy = random.uniform(
            self.min_stored_energy, self.max_stored_energy
        )

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
            self.current_date,
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

    def _set_state_iter(self, price_data: dict[datetime, float]):
        self._state_iter = iter(price_data.items())

    def _set_next_state(self):
        try:
            self.current_date, self.current_price = next(self._state_iter)
        except StopIteration:
            self.terminated = True

    def _get_state(self):
        return (
            self.current_date.hour,
            self.current_date.month - 1,
            self._get_price_bin(),
            self._get_reservoir_bin(),
        )

    def _get_price_bin(self):
        return int(self.current_price // self.price_bin_size)

    def _get_reservoir_bin(self):
        return int(
            self.stored_energy // (self.max_stored_energy / self.n_bins_reservoir)
        )

    def _get_reward(self, flow: float):
        # positive flow = selling
        if flow > 0:
            return flow * self.sell_multiplier * self.current_price

        # negative flow = buying
        return flow * self.buy_multiplier * self.current_price
