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

    return df["value"].to_dict()

def joule_to_kwh(joule: float):
    return joule / 3.6e6

def kwh_to_joule(kwh: float):
    return kwh * 3.6e6