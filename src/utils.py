from datetime import datetime

import numpy as np
import pandas as pd


# convert dateframe
def convert_dataframe(df: pd.DataFrame) -> dict[datetime, float]:
    df = (
        df.rename(columns={"PRICES": "date"})
        .set_index("date")
        .rename(columns=lambda col: col.lstrip("Hour "))
        .melt(ignore_index=False)
        .reset_index()
    )

    df["variable"] = df["variable"].astype(int) - 1  # -1 because 1 corresponds to 00:00
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(
        df["variable"], unit="h"
    )
    df.drop(columns=["date", "variable"], inplace=True)
    df.set_index("datetime", inplace=True)

    return df["value"].to_dict()


def joule_to_mwh(joule: float):
    return joule / 3.6e9


def cumsum(data: list[float]):
    data_sum = 0.0
    res = []

    for value in data:
        data_sum += value
        res.append(data_sum)

    return res


def add_range_prices(dict: dict, low, medium):
    # df = convert_dataframe(df_original)
    df = [*dict.values()]
    df = np.sort(df)

    low, medium, high = (
        df[0 : int((low * len(df)))],
        df[int((low * len(df))) : int(((medium + low) * len(df)))],
        df[int((medium + low) * len(df)) :],
    )
    low_min_max, medium_min_max, high_min_max = (
        (low[0], low[-1]),
        (medium[0], medium[-1]),
        (high[0], high[-1]),
    )

    return low_min_max, medium_min_max, high_min_max


def plt_col(lst):
    cols = []
    for l in lst:
        if l == 0:
            cols.append("green")
        elif l == 1:
            cols.append("blue")
        else:
            cols.append("red")
    return cols

def discounted_reward(rewards:list[float], gamma:float):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0.0

    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add

    return discounted_rewards