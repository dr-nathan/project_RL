import pandas as pd

from src.baseline_stupid import Baseline
from src.environment import DiscreteDamEnv
import src.utils

train = pd.read_excel("./data/train.xlsx")
train_dict = src.utils.convert_dataframe(train)

val = pd.read_excel("./data/validate.xlsx")
val_dict = src.utils.convert_dataframe(val)
val_env = DiscreteDamEnv(val_dict)

a = Baseline(env=val_env, df=train)
a.fit(train_dict, low_perc=0.25, medium_perc=0.5)
a.plot_baseline()
