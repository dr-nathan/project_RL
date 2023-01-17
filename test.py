# %%
from datetime import datetime
import random
import pandas as pd
from src.utils import convert_dataframe

df = pd.read_excel("data/validate.xlsx")

lookup = convert_dataframe(df)
lookup[datetime(2010, 1, 1, 0, 0)]


# %%
from src.environment import DiscreteDamEnv

env = DiscreteDamEnv(lookup)

# %%
env.step(1)

# %%
env.reset()
run = True
while run:
    # res = env.step(random.choice([0, 1, 2]))
    res = env.step(2)
    run = not res[2]
    # print(res[0], round(res[1], 2), env.stored_energy, env.current_price)

# %%
env.episode_data.debug_plot()
