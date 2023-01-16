import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tsfresh






def plot_vol(prices_train):
    TRADING_DAYS= 365*24
    returns = np.log(prices_train/prices_train.shift(1))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
    volatility.tail()
    print(len(volatility))
    #print(volatility)
    nan_count = volatility.isna().sum()
    print(nan_count)
    diff_vol_years = volatility[TRADING_DAYS:TRADING_DAYS+TRADING_DAYS] - volatility[TRADING_DAYS+TRADING_DAYS:len(volatility)]
    #print(diff_vol_years)
    print(len(volatility[TRADING_DAYS:TRADING_DAYS+TRADING_DAYS] ))
    print(len(volatility[TRADING_DAYS+TRADING_DAYS:len(volatility)]))
    #print(diff_vol_years.isna().sum())
    #print(volatility[365:365+365])
    #print( volatility[TRADING_DAYS+TRADING_DAYS:len(volatility)])
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.plot(volatility, label = "volatility")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volatility')
    ax1.set_title('Annualized volatility for Apple Inc')
    plt.legend()
    plt.show()







def plot(x,mean, std,what):
    sns.set()
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    axs.plot(x,mean,label='Average hour price')
    axs.errorbar(x, mean,std,  linestyle='None', marker='+')
    axs.set_xlabel(what)
    axs.set_ylabel('Price')
    axs.set_title(f"Average {what} price")




    fig.tight_layout()
    plt.show()





def plot_hour_month(b):
    plt.plot(b.get_group('01').groupby(['hour'])['price'].mean(),label='Jan')
    plt.plot(b.get_group('02').groupby(['hour'])['price'].mean(),label='Feb')
    plt.plot(b.get_group('03').groupby(['hour'])['price'].mean(),label='March')
    plt.plot(b.get_group('04').groupby(['hour'])['price'].mean(),label='April')
    plt.plot(b.get_group('05').groupby(['hour'])['price'].mean(),label='May')
    plt.plot(b.get_group('06').groupby(['hour'])['price'].mean(),label='June')
    plt.plot(b.get_group('07').groupby(['hour'])['price'].mean(),label='July')
    plt.plot(b.get_group('08').groupby(['hour'])['price'].mean(),label='August')
    plt.plot(b.get_group('09').groupby(['hour'])['price'].mean(),label='September')
    plt.plot(b.get_group('10').groupby(['hour'])['price'].mean(),label='October')
    plt.plot(b.get_group('11').groupby(['hour'])['price'].mean(),label='November')
    plt.plot(b.get_group('12').groupby(['hour'])['price'].mean(),label='December')
    plt.legend()
    plt.show()


def plot_hour_day(a):
    plt.plot(a.get_group(0).groupby(['hour'])['price'].mean(),label='Monday')
    plt.plot(a.get_group(1).groupby(['hour'])['price'].mean(),label='Tuesday')
    plt.plot(a.get_group(2).groupby(['hour'])['price'].mean(),label='Wednesday')
    plt.plot(a.get_group(3).groupby(['hour'])['price'].mean(),label='Thursday')
    plt.plot(a.get_group(4).groupby(['hour'])['price'].mean(),label='Friday')
    plt.plot(a.get_group(5).groupby(['hour'])['price'].mean(),label='Saturday')
    plt.plot(a.get_group(6).groupby(['hour'])['price'].mean(),label='Sunday')
    plt.legend()
    plt.show()





def main():
    df = pd.read_excel("./data/train.xlsx")
    dict = convert_dataframe(df)
    prices_train_list = [*dict.values()]
    prices_train =  pd.DataFrame(prices_train_list)

    df_new = pd.DataFrame()
    df_new['time'] = [*dict.keys()]
    df_new['week_day'] = [i.weekday() for i in df_new['time']]
    df_new['month'] = df_new['time'].dt.strftime('%m')
    df_new['hour'] = df_new['time'].dt.strftime('%H')
    df_new['price'] = [*dict.values()]
    
    
    #a = pd.DataFrame()
    #a['time'] =df_new['time']
    #a['price']=df_new['price']
    #features = tsfresh.extract_features(a, column_id="time")
    #features.to_csv('features.csv')
    #tsfresh.utilities.dataframe_functions.impute(features)
    #feat = pd.read_csv('./features.csv')
    #print(feat['price__variance'])





    m = df_new.groupby(['week_day'])['price'].mean()
    print(m)
    s = df_new.groupby(['week_day'])['price'].std()
    x = np.linspace(0, 7, 7)  
    #mon = s.get_group(0)
    #plot(x,m,s,'weekday')


    m = df_new.groupby(['month'])['price'].mean()
    s = df_new.groupby(['month'])['price'].std()
    x = np.linspace(0, 12, 12)  
    #plot(x,m,s,'month')

    m = df_new.groupby(['hour'])['price'].mean()
    s = df_new.groupby(['hour'])['price'].std()
    x = np.linspace(0, 24, 24)  
    #plot(x,m,s,'month')

    a = df_new.groupby(['week_day'])
    plot_hour_day(a)
    
    b = df_new.groupby(['month'])
    plot_hour_month(b)



    










 



if __name__ == "__main__":
    main()    