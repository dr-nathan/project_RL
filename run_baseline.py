import pandas as pd

from src.baseline_stupid import Baseline
from src.environment import DiscreteDamEnv, DamEpisodeData
import src.utils

df = pd.read_excel("./data/train.xlsx")
val = pd.read_excel("./data/validate.xlsx")
dict = src.utils.convert_dataframe(df)
env = DiscreteDamEnv(dict)
a = Baseline(env = env, df=df, low_perc=0.25, medium_perc=0.5, val=val)  #
a.plot_baseline()
