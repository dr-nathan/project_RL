# %%
from datetime import datetime
import pandas as pd
from src.utils import convert_dataframe

df = pd.read_excel("data/validate.xlsx")

lookup = convert_dataframe(df)
lookup[datetime(2010, 1, 1, 0, 0)]
