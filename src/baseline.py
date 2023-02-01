from datetime import datetime
import numpy as np
import pandas as pd

from src.environment.dam import DiscreteDamEnv
from src.utils import convert_dataframe


class Baseline:
    def __init__(
        self,
        env: DiscreteDamEnv,
        df: pd.DataFrame,
    ):
        self.env = env

        dict_val = convert_dataframe(df)
        self.prices_val = [*dict_val.values()]
        self.prices = np.sort(
            self.prices_val
        )  # sort prices overall - maybe could do that per month or year?? I sort them such that then I can take the percentages

    def fit(
        self, training_data: dict[datetime, float], low_perc: float, medium_perc: float
    ):
        self.low_perc = low_perc
        self.medium_perc = medium_perc

        self.prices = [*training_data.values()]
        self.prices = np.sort(self.prices)
        (
            self.low_min_max,
            self.medium_min_max,
            self.high_min_max,
        ) = self.get_low_medium_high_price()

    def get_low_medium_high_price(self):
        # divide prices in low medium high according to given percentage

        low, medium, high = (
            self.prices[0 : int((self.low_perc * len(self.prices)))],
            self.prices[
                int((self.low_perc * len(self.prices))) : int(
                    ((self.medium_perc + self.low_perc) * len(self.prices))
                )
            ],
            self.prices[int((self.medium_perc + self.low_perc) * len(self.prices)) :],
        )

        self.low_min_max, self.medium_min_max, self.high_min_max = (
            (low[0], low[-1]),
            (medium[0], medium[-1]),
            (high[0], high[-1]),
        )

        return self.low_min_max, self.medium_min_max, self.high_min_max

    def _select_action_threshold(self):
        current_price = self.env.current_price

        # buy
        if current_price <= self.low_min_max[1]:
            return 2

        # sell
        if self.high_min_max[0] <= current_price:
            return 1

        # else, do nothing
        return 0

    def _select_action_hourly(self):
        current_hour = self.env.current_date.hour

        # buy
        if 2 <= current_hour <= 6:
            return 2

        # sell
        if 9 <= current_hour <= 19:
            return 1

        # else, do nothing
        return 0

    def run(
        self,
        strategy: str,
        plot: bool = True,
        plot_title: str | None = None,
    ):
        if strategy == "threshold":
            action_selection = self._select_action_threshold

        elif strategy == "hourly":
            action_selection = self._select_action_hourly

        else:
            raise ValueError(f"Strategy {strategy} not implemented")

        self.env.reset()
        terminated = False

        while not terminated:
            action = action_selection()
            _, _, terminated, *_ = self.env.step(action)

        if plot:
            self.env.episode_data.debug_plot(plot_title)
            self.env.episode_data.plot_fancy()
