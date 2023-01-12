from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class baseline():
    max_stored_energy = joule_to_kwh(100000 * 1000 * 9.81 * 30)  # U = mgh
    min_stored_energy = 0
    max_flow_rate = joule_to_kwh(5 * 3600 * 9.81 * 30)  # 5 m^3/s to m^3/h * gh
    buy_multiplier = 1.2  # i.e. we spend 1.2 Kw to store 1 Kw (80% efficiency)
    sell_multiplier = 0.9  # i.e. we get 0.9 Kw for selling 1 Kw (90% efficiency)
    
    def __init__(self, df: pd.DataFrame, low_perc: np.float32,medium_perc: np.float32, val:pd.DataFrame ):
        self.low_perc = low_perc
        self.medium_perc = medium_perc

        dict = convert_dataframe(df)
        self.prices_train = [*dict.values()]
        self.prices = np.sort(self.prices_train) #sort prices overall - maybe could do that per month or year?? I sort them such that then I can take the percentages

        dict_val = convert_dataframe(val) #convert the validation data as well to try the heuristic
        self.prices_val = [*dict_val.values()]

    def get_low_medium_high_price(self):
        #divide prices in low medium high according to given percentage

        low,medium,high = self.prices[0:int((self.low_perc*len(self.prices)))], self.prices[int((self.low_perc*len(self.prices))):int(((self.medium_perc+self.low_perc)*len(self.prices)))], self.prices[int((self.medium_perc+self.low_perc)*len(self.prices)):]
        self.low_min_max, self.medium_min_max, self.high_min_max= (low[0],low[-1]), (medium[0],medium[-1]), (high[0],high[-1])
        
        return self.low_min_max, self.medium_min_max, self.high_min_max

    def choice(self) :
        reward = [] #initialize reward
        energy = self.max_stored_energy
        energy_story = [energy] 
        self.low_min_max, self.medium_min_max, self.high_min_max = self.get_low_medium_high_price()

        #for i in self.prices_train: #go through prices and takes decision
        for i in self.prices_val:    
            if i >= self.low_min_max[0] and i <= self.low_min_max[1] and energy < self.max_stored_energy : #if ith price in low range and there is space to store - buy 
                reward.append(-i*self.buy_multiplier) #if we buy: -1 * unit price * buy mult
                energy = energy + self.max_flow_rate
                energy_story.append(energy)
                
            if i > self.medium_min_max[0] and i <= self.medium_min_max[1] : #if ith price in medium range dont do anything
                reward.append(0)  #we are IGNAVI: nothing happens to the reward
                energy = energy #nothing happens to energy
                energy_story.append(energy) 

            if i > self.high_min_max[0] and i <= self.high_min_max[1] and energy > 0: ##if ith price in high range and there is energy to sell - sell
                reward.append(i*self.sell_multiplier) #if we sell: 1* unit price * sell mult
                energy = energy - self.max_flow_rate
                energy_story.append(energy)

        print('total reward??',np.sum(reward))  #this is total reward on validation set
        return energy_story,reward


    def plot(self):
        energy_story,reward = self.choice()
        sns.set()
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        axs[0].plot(energy_story)
        axs[0].set_title("Energy over time")

        axs[1].plot(reward)
        axs[1].set_title("Reward over time")

        plt.show()

        


df = pd.read_excel('./data/train.xlsx')
val = pd.read_excel('./data/validate.xlsx')
a = baseline(df=df,low_perc=0.2,medium_perc=0.07,val=val) #
a.plot()


