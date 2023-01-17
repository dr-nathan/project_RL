from datetime import datetime
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

    # keep track of real price
    df_real = df.copy(deep=True)

    # calculate upper 99% quantile
    upper_99 = df["value"].quantile(0.99)
    #replace values above upper 99% quantile with upper 99% quantile
    df.loc[df["value"] > upper_99, "value"] = upper_99
    # TODO: val data should use train data quantile

    return df["value"].to_dict(), df_real["value"].to_dict()


def joule_to_mwh(joule: float):
    return joule / 3.6e9


def cumsum(data: list[float]):
    data_sum = 0.0
    res = []

    for value in data:
        data_sum += value
        res.append(data_sum)

    return res
