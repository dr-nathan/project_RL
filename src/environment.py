from datetime import datetime, timedelta
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from .utils import joule_to_kwh


class DiscreteDamEnv(gym.Env):
    """Dam Environment that follows gym interface"""

    # static properties
    max_stored_energy = joule_to_kwh(100000 * 1000 * 9.81 * 30)  # U = mgh
    min_stored_energy = 0
    # a positive flow means emtpying the reservoir
    max_flow_rate = joule_to_kwh(5 * 3600 * 9.81 * 30)  # 5 m^3/s to m^3/h * gh

    buy_multiplier = 1.2  # i.e. we spend 1.2 Kw to store 1 Kw (80% efficiency)
    sell_multiplier = 0.9  # i.e. we get 0.9 Kw for selling 1 Kw (90% efficiency)

    price_bin_size = 100

    def __init__(self, price_data: dict[datetime, float]):
        super().__init__()

        self.price_data = price_data

        self.reset()

        # 0 = do nothing
        # 1 = empty / sell
        # 2 = fill / buy
        self.action_space = spaces.Discrete(3)

        # state is (hour, electricity price (bins))
        n_bins = int(max(self.price_data.values()) // self.price_bin_size)
        self.observation_space = spaces.MultiDiscrete([24, n_bins])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self.stored_energy = (
            self.max_stored_energy / 2
        )  # start with a half-full reservor
        self.current_date = min(self.price_data.keys())
        self.current_price = self.price_data[self.current_date]

        self.terminated = False

        return self._get_state()

    def step(self, action: int):
        # empty
        if action == 1:
            flow_rate = self.max_flow_rate

        # fill
        elif action == 2:
            flow_rate = -self.max_flow_rate

        # do nothing, i.e. action == 0 (default)
        else:
            flow_rate = 0

        # update the applied flow so we don't overflow or store less than 0
        applied_flow_rate = self._constrain_flow_rate(flow_rate)

        # move to the next hour
        self._set_next_state()

        # observation, reward, terminated, info (Gym convention)
        return (
            self._get_state(),
            self._get_reward(applied_flow_rate),
            self.terminated,
            {},
        )

    def _constrain_flow_rate(self, flow_rate: float):
        self.stored_energy += flow_rate

        # change flow rate if we overflow
        if self.stored_energy > self.max_stored_energy:
            correction = self.stored_energy - self.max_stored_energy
            flow_rate -= correction
            self.stored_energy = self.max_stored_energy

        # change flow rate if we store less than 0
        elif self.stored_energy < self.min_stored_energy:
            correction = self.min_stored_energy - self.stored_energy
            flow_rate += correction
            self.stored_energy = self.min_stored_energy

        return flow_rate

    def _set_next_state(self):
        self.current_date += timedelta(hours=1)

        if self.current_date in self.price_data:
            self.current_price = self.price_data[self.current_date]

        else:
            self.terminated = True

    def _get_state(self):
        return [self.current_date.hour, self._get_price_bin()]

    def _get_price_bin(self):
        return self.current_price // self.price_bin_size

    def _get_reward(self, flow: float):
        # positive flow = selling, negative = buying
        if flow > 0:
            return flow * self.sell_multiplier * self.current_price

        return flow * self.buy_multiplier * self.current_price
