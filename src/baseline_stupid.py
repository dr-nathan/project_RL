from datetime import datetime
import numpy as np
import pandas as pd

from src.environment import DiscreteDamEnv
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

    def choice(self):
        self.env.reset()

        # go through prices and takes decision
        terminated = False
        while not terminated:
            current_price = self.env.current_price
            # BUY
            if current_price <= self.low_min_max[1]:
                action = 2

                _, _, terminated, *_ = self.env.step(action)

            # NOTHING
            elif self.medium_min_max[0] < current_price < self.medium_min_max[1]:
                action = 0
                _, _, terminated, *_ = self.env.step(action)

            # SELL
            elif self.high_min_max[0] <= current_price:
                action = 1
                _, _, terminated, *_ = self.env.step(action)

        print("total reward", np.sum(self.env.episode_data.reward))

        return self.env.episode_data

    def choice2(self):
        self.env.reset()

        terminated = False
        while not terminated:
            curr_date = self.env.current_date

            current_hour = curr_date.hour

            # buy
            if 2 <= current_hour <= 6:
                action = 2
                _, _, terminated, *_ = self.env.step(action)
            # sell
            elif 9 <= current_hour <= 19:
                action = 1
                _, _, terminated, *_ = self.env.step(action)
            # do nothing
            else:
                action = 0
                _, _, terminated, *_ = self.env.step(action)

        print(
            "total reward??", np.sum(self.env.episode_data.reward)
        )  # this is total reward on validation set

        return self.env.episode_data

    def run(self, strategy: str, plot: bool = True, **kwargs):
        if strategy == "threshold":
            data = self.choice()
        elif strategy == "hourly":
            data = self.choice2()
        else:
            raise ValueError(f"Strategy {strategy} not implemented")

        if plot:
            data.plot(**kwargs)
