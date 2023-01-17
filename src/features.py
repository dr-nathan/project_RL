import pandas as pd
import numpy as np
from src.utils import *
import matplotlib.pyplot as plt
import seaborn as sns






def plot_vol(prices_train):
    TRADING_DAYS= 365*24
    returns = np.log(prices_train/prices_train.shift(1))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
    volatility.tail()
    #print(len(volatility))
    #print(volatility)
    nan_count = volatility.isna().sum()
    #print(nan_count)
    diff_vol_years = volatility[TRADING_DAYS:TRADING_DAYS+TRADING_DAYS] - volatility[TRADING_DAYS+TRADING_DAYS:len(volatility)]
    #print(diff_vol_years)
    #print(len(volatility[TRADING_DAYS:TRADING_DAYS+TRADING_DAYS] ))
    #print(len(volatility[TRADING_DAYS+TRADING_DAYS:len(volatility)]))
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


def plot_year(a):
    plt.plot(a.get_group('2007').groupby(['hour'])['price'].mean(),label='2007')
    plt.plot(a.get_group('2008').groupby(['hour'])['price'].mean(),label='2008')
    plt.plot(a.get_group('2009').groupby(['hour'])['price'].mean(),label='2009')
    plt.legend()
    plt.show()


def season(df):
    
    if df['month'] in ['01' ,'02' ,'03']:
        return 1
    elif df['month'] in ['04' ,'05' , '06']:
        return 2
    elif df['month'] in ['07' , '08' , '09']:
        return 3
    elif df['month'] in [ '10' , '11' , '12']:
        return 4    

def nigh_day(df):
    
    if df['hour'] in ['00' ,'02' ,'03','04','05', '22','23']:
        return 0
    elif df['hour'] in ['06' ,'07' , '08','09','10','11','12','13','14','15','16','17','18']:
        return 1
    elif df['hour'] in ['19' , '20' , '21']:
        return 2


       


def plot_hour_season(a):
    plt.plot(a.get_group(1).groupby(['hour'])['price'].mean(),label='Winter')
    plt.plot(a.get_group(2).groupby(['hour'])['price'].mean(),label='Spring')
    plt.plot(a.get_group(3).groupby(['hour'])['price'].mean(),label='Summer')
    plt.plot(a.get_group(4).groupby(['hour'])['price'].mean(),label='Autumn')

    plt.legend()
    plt.show()


def get_low_medium_high_price(df,low_perc=0.2,medium_perc=0.7):
    #divide prices in low medium high according to given percentage
    #df = df['price']
    df = list(df)
    
    low,medium,high = df[0:int((low_perc*len(df)))], df[int((low_perc*len(df))):int(((medium_perc+low_perc)*len(df)))], df[int((medium_perc+low_perc)*len(df)):]
    low_min_max, medium_min_max, high_min_max= (low[0],low[-1]), (medium[0],medium[-1]), (high[0],high[-1])

    for i in df:   

        if i >= low_min_max[0] and i <= low_min_max[1] :
            return 0
        if i > medium_min_max[0] and i <= medium_min_max[1] :   
            return 1
        else:
            return 2

        
    #return low_min_max, medium_min_max, high_min_max   
    #  

def create_df(df):
    dict = convert_dataframe(df)
    df_new = pd.DataFrame()
    df_new['time'] = [*dict.keys()]
    df_new['price'] = [*dict.values()]
    df_new['week_day'] = [i.weekday() for i in df_new['time']]
    df_new['month'] = df_new['time'].dt.strftime('%m')
    df_new['hour'] = df_new['time'].dt.strftime('%H')
    df_new['season'] =  df_new.apply(season, axis=1)
    df_new['price_range'] = get_low_medium_high_price(df_new['price'])
    df_new['night_day'] = df_new.apply(nigh_day,axis=1)
    df_new['year'] = df_new['time'].dt.strftime('%Y')

    return df_new



def main():
    df = pd.read_excel("./data/train.xlsx")
    df_new = create_df(df)
  
    #dict = convert_dataframe(df)
    #prices_train_list = [*dict.values()]
    #prices_train =  pd.DataFrame(prices_train_list)

    #df_new = pd.DataFrame()
    #df_new['time'] = [*dict.keys()]
    #df_new['price'] = [*dict.values()]
    #df_new['week_day'] = [i.weekday() for i in df_new['time']]
    #df_new['month'] = df_new['time'].dt.strftime('%m')
    #df_new['hour'] = df_new['time'].dt.strftime('%H')
    
    #df_new['season'] =  df_new.apply(season, axis=1)
    #df_new['price_range'] = get_low_medium_high_price(df_new['price'])
    #print(df_new)
    
    #a = pd.DataFrame()
    #a['time'] =df_new['time']
    #a['price']=df_new['price']
    #features = tsfresh.extract_features(a, column_id="time")
    #features.to_csv('features.csv')
    #tsfresh.utilities.dataframe_functions.impute(features)
    #feat = pd.read_csv('./features.csv')
    #print(feat['price__variance'])





    m = df_new.groupby(['week_day'])['price'].mean()
    s = df_new.groupby(['week_day'])['price'].std()
    x = np.linspace(0, 7, 7)  
    #plot(x,m,s,'weekday')


    m = df_new.groupby(['month'])['price'].mean()
    s = df_new.groupby(['month'])['price'].std()
    x = np.linspace(0, 12, 12)  
    #plot(x,m,s,'month')

    m = df_new.groupby(['hour'])['price'].mean()
    s = df_new.groupby(['hour'])['price'].std()
    x = np.linspace(0, 24, 24)  
    #plot(x,m,s,'hour')

 

    a = df_new.groupby(['week_day'])
    plot_hour_day(a)
    print(a.get_group(0).groupby(['hour'])['price'].mean())

    b = df_new.groupby(['month'])
    plot_hour_month(b)

    c = df_new.groupby(['season'])
    plot_hour_season(c)

    d = df_new.groupby(['year'])
    plot_year(d)

    plt.plot(df_new['price'])
    plt.show()



    










 



if __name__ == "__main__":
    main()    