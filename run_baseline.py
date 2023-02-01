import pandas as pd

from src.baseline import Baseline
from src.environment.dam import DiscreteDamEnv
import src.utils

if __name__ == "__main__":
    train = pd.read_excel("./data/train.xlsx")
    train_dict = src.utils.convert_dataframe(train)

    val = pd.read_excel("./data/validate.xlsx")
    val_dict = src.utils.convert_dataframe(val)
    val_env = DiscreteDamEnv(val_dict)

    a = Baseline(env=val_env, df=train)
    a.fit(train_dict, low_perc=0.4, medium_perc=0.2)
    a.run(plot_title="Validation Threshold Baseline", strategy="threshold", plot=True)

    a.run(plot_title="Validation Hourly Baseline", strategy="hourly", plot=True)
