import pandas as pd

from src.baseline_stupid import Baseline

df = pd.read_excel("./data/train.xlsx")
val = pd.read_excel("./data/validate.xlsx")
a = Baseline(df=df, low_perc=0.2, medium_perc=0.07, val=val)  #
a.plot_baseline()
