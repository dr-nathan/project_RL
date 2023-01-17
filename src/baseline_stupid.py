import gym
from src.utils import *
import pandas as pd
import numpy as np
from src.environment import DamEpisodeData, DiscreteDamEnv
from src.features import *


class Baseline:
    def __init__(
        self,
        env: DiscreteDamEnv,
        df: pd.DataFrame,
        low_perc: np.float32,
        medium_perc: np.float32,
        val: pd.DataFrame,
    ):
        self.low_perc = low_perc
        self.medium_perc = medium_perc
        self.env = env

        self.max_stored_energy = env.max_stored_energy
        self.min_stored_energy = env.min_stored_energy
        self.max_flow_rate = env.max_flow_rate
        self.buy_multiplier = env.buy_multiplier
        self.sell_multiplier = env.sell_multiplier

        self.df_new = create_df(df)

        dict_val = convert_dataframe(df)
        self.prices_val = [*dict_val.values()]
        self.date =  [*dict_val.keys()]
        self.prices = np.sort(
            self.prices_val
        )  # sort prices overall - maybe could do that per month or year?? I sort them such that then I can take the percentages

        dict_val = convert_dataframe(
            val
        )  # convert the validation data as well to try the heuristic


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
        energy = self.env.stored_energy
        (
            self.low_min_max,
            self.medium_min_max,
            self.high_min_max,
        ) = self.get_low_medium_high_price()
        #go through prices and takes decision
        for i in self.prices_val:
            # BUY
            if (
                i >= self.low_min_max[0]
                and i <= self.low_min_max[1]
                and energy < self.max_stored_energy
            ): 
                action = 2

                self.env.step(action)

            # NOTHING
            if (
                i > self.medium_min_max[0] and i < self.medium_min_max[1]
            ):  
                action = 0
                self.env.step(action)

            # SELL
            if (
                i >= self.high_min_max[0] and i <= self.high_min_max[1] and energy > 0
            ): 
                action = 1
                self.env.step(action)

        print(
            "total reward", np.sum(self.env.episode_data.reward)
        ) 
        return self.env.episode_data

    def choice2(self):
        energy = self.env.stored_energy
        for i in self.date:
            i = i.strftime('%H')
            
            if i in ['00' ,'01','02' ,'03','04','05', '22','23']  and energy < self.max_stored_energy :
                action = 2
                self.env.step(action)
            if i in ['06' ,'07' , '08','09','10','11','12','13','14','15','16','17','18'] and energy > 0:
                action = 1
                self.env.step(action)
            if i in ['19' , '20' , '21']:
                action = 0    
                self.env.step(action)    
        print(
            "total reward??", np.sum(self.env.episode_data.reward)
        )  # this is total reward on validation set
        return self.env.episode_data   

    def choice3(self):
        (
            self.low_min_max,
            self.medium_min_max,
            self.high_min_max,
        ) = self.get_low_medium_high_price()
        energy = self.env.stored_energy
        for i,j in zip(self.date,self.prices):
  
            if  (i.strftime('%H') in ['00' ,'02' ,'03','04','05', '22','23'] and energy < self.max_stored_energy or i.weekday() in ['6'] and energy < self.max_stored_energy ) :
                action=2
                self.env.step(action)
            
            if (i.strftime('%H') in [ '06' ,'07' ,'08','09','10','11','12','13','14','15','16','17','18'] ) and energy > 0:
                action=1   
                self.env.step(action) 
            else:
                action=0
                self.env.step(action) 

        print(
            "total reward??", np.sum(self.env.episode_data.reward)
        )  # this is total reward on validation set
        return self.env.episode_data          



    def plot_baseline(self):

        data = self.choice2()  # storage is in terms of energy here
        data.plot()


