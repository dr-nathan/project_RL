import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.utils import cumsum, joule_to_mwh, add_range_prices, plt_col


@dataclass
class DamEpisodeData:
    """Dataclass to store episode data for a dam environment"""

    date: list[datetime] = field(default_factory=list)
    storage: list[float] = field(default_factory=list)
    action: list[float] = field(default_factory=list)
    flow: list[float] = field(default_factory=list)
    price: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    reward_cumulative = property(lambda self: cumsum(self.reward))
    total_reward = property(lambda self: sum(self.reward))

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

    def debug_plot(self, title: str | None = None):
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

        axs[5].plot(self.date, self.reward_cumulative)
        axs[5].set_title("Cumulative reward")

        fig.tight_layout()
        plt.show()

    def plot_fancy(self):
        sns.set()
        price = self.price[-1001:-1]
        action = self.action[-1000:]
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        cols = plt_col(action)

        df = pd.DataFrame({"price": price, "action": action}).reset_index()
        df.action = df.action.map({0: "nothing", 1: "sell", 2: "buy"})

        sns.scatterplot(
            data=df,
            x="index",
            y="price",
            hue="action",
            palette={"nothing": "blue", "sell": "green", "buy": "red"},
        )
        plt.ylim(0, 170)
        plt.title("Action on the prices over time")

        # axs.scatter(range(len(price)),price,s=100, c=cols,marker= 'o', label=cols)
        # axs.legend()
        # axs.set_title("Action on the prices")
        # fig.tight_layout()
        plt.show()


class DamEnvBase(gym.Env):
    """Dam Environment that follows gym interface"""

    # static properties
    max_stored_energy = joule_to_mwh(
        100000 * 1000 * 9.81 * 30
    )  # 100000 m^3 to mwh with U = mgh
    min_stored_energy = 0
    # a positive flow means emtpying the reservoir
    max_flow_rate = joule_to_mwh(5 * 1000 * 3600 * 9.81 * 30)  # 5 m^3/s to mwh

    buy_multiplier = 1.25  # i.e. we spend 1.25 Kw to store 1 Kw (80% efficiency)
    sell_multiplier = 0.9  # i.e. we get 0.9 Kw for selling 1 Kw (90% efficiency)

    def __init__(self, price_data: dict[datetime, float]):
        super().__init__()

        self.price_data = dict(sorted(price_data.items()))

        self.max_price = max(self.price_data.values())

        # self.action_space and self.observation_space must be set in subclasses

    def reset(
        self,
        *,
        seed: int | None = None,
        start_amount: float | Literal["random"] = 0.5,
        random_startpoint: bool = False,
        price_data: dict[datetime, float] | None = None,
    ):
        super().reset(seed=seed)

        if price_data:
            self.price_data = dict(sorted(price_data.items()))
        else:
            assert (
                self.price_data and len(self.price_data) > 0
            ), "No price data provided"

        # reservoir starting level
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
            return self._pick_random_startpoint()

        return self._get_state()

    def _pick_random_startpoint(self):
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
        flow_rate = self._action_to_flow(action)

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

    def _get_reward(self, flow: float):
        # positive flow = selling
        if flow > 0:
            return flow * self.sell_multiplier * self.current_price

        # negative flow = buying
        return flow * self.buy_multiplier * self.current_price

    def _action_to_flow(self, action: int | float):
        raise NotImplementedError("Must be implemented in subclass")

    def _get_state(self):
        raise NotImplementedError("Must be implemented in subclass")


class DiscreteDamEnv(DamEnvBase):
    n_bins_price = 20
    n_bins_reservoir = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # have price bins split into equal quantiles
        self.quantiles = np.quantile(
            list(self.price_data.values()), np.linspace(0, 1, self.n_bins_price + 1)
        )[1:-1]

        # 0 = do nothing
        # 1 = empty / sell
        # 2 = fill / buy
        self.action_space = spaces.Discrete(3)

        # state is (hour, electricity price (bins), stored energy (bins))
        self.observation_space = spaces.MultiDiscrete(
            [24, self.n_bins_price, self.n_bins_reservoir]
        )

    def _action_to_flow(self, action: int):
        # empty reservor / sell
        if action == 1:
            return self.max_flow_rate

        # fill reservor / buy
        if action == 2:
            return -self.max_flow_rate

        # do nothing, i.e. action == 0 (default)
        return 0.0

    def _get_state(self):
        return (
            self.current_date.hour,
            self._get_price_bin(),
            self._get_reservoir_bin(),
        )

    def _get_price_bin(self):
        # get bins with equal number of data points
        return np.searchsorted(
            self.quantiles, self.current_price
        )  # much quicker than np.digitize

    def _get_reservoir_bin(self):
        return int(
            self.stored_energy // (self.max_stored_energy / self.n_bins_reservoir)
        )


class ContinuousDamEnv(DamEnvBase):
    def __init__(self, *args, **kwargs):
        # action is the flow rate
        self.action_space = spaces.Box(low=-1, high=1)

        # state is (hour, electricity price, stored energy)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,))

        super().__init__(*args, **kwargs)

    def _action_to_flow(self, action: float):
        return action * self.max_flow_rate

    def _get_state(self):
        return (
            self.current_date.hour / 24,
            self.current_price / self.max_price,
            self.stored_energy / self.max_stored_energy,
            self._is_winter(),
        )

    def _is_winter(self):
        # November is actually not winter, but we generally see higher prices here too
        return self.current_date.month in [1, 2, 12, 11]

class DiscreteContinuousDamEnv(ContinuousDamEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Discrete(3)

    def _action_to_flow(self, action: int):
        # empty reservor / sell
        if action == 1:
            return self.max_flow_rate

        # fill reservor / buy
        if action == 2:
            return -self.max_flow_rate

        # do nothing, i.e. action == 0 (default)
        return 0.0
