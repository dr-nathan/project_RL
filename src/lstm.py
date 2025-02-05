from locale import normalize
from turtle import forward
from attr import dataclass
import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

import random
from tqdm import tqdm
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler,StandardScaler








class LSTM(nn.Module):
    
    def __init__(self,input_size = 1, hidden_size = 100, out_size = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size,out_size)
        self.hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
    
    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq),-1))
        return pred[-1]


class LSTM_price():
    def __init__(self):
        time = 365*24
        self.time = time
        self.window_size = 12
        self.future = 24
        self.epochs = 10
        self.trained_model_name = 'trained_lstm_new_exp'

    def input_data(self,seq,ws):
        out = []
        L = len(seq)
        for i in range(L-ws):
            window = seq[i:i+ws]
            label = seq[i+ws:i+ws+1]
            out.append((window,label))
        return out


    def import_process_data(self,path):

        df = pd.read_excel(path)
        dicti = convert_dataframe(df)

        price_data = dict(sorted(dicti.items()))
        values_array = np.array(list(price_data.values()))
        #values_array = preprocessing.normalize([values_array])
        values = ((torch.FloatTensor(values_array)))
        values = nn.functional.normalize(values,dim=0)
        self.max_values = torch.max(values)

        train_set = values
        train_set = values[:self.time]
        test_set = values[self.time:self.time+30*24]


        self.train_set = train_set
        
        self.test_set = test_set
    

        return test_set,train_set



 


    def training_loop(self,train_data,epochs):
        torch.manual_seed(42)
        model = LSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        self.epochs = epochs


        for epoch in tqdm(range(self.epochs)):
            for seq, y_train in train_data:
                optimizer.zero_grad()
                model.hidden = (torch.zeros(1,1,model.hidden_size),
                            torch.zeros(1,1,model.hidden_size))
                
                y_pred = model(seq)


                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
                
            print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
        
        torch.save(model.state_dict(), self.trained_model_name)  
        return model


    def train(self):
        window_size = 24*30
        test_set,train_set = self.import_process_data("./data/train.xlsx")
        #test_set,train_set = self.normalize()
        train_data = self.input_data(train_set, self.window_size)
        
        model = self.training_loop(train_data,self.epochs)
        
        criterion = nn.MSELoss()
        preds = train_set[-window_size:].tolist()
        model.eval()
        for f in range(self.future):
            seq = torch.FloatTensor(preds[-window_size:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1,1,model.hidden_size),
                            torch.zeros(1,1,model.hidden_size))
                preds.append(model(seq).item())
            
        loss = criterion(torch.tensor(preds[-window_size:]), train_set[-window_size:])
        print('prediction',preds[window_size:])
        print('real',train_set[-window_size:])
        print(f"Performance on test range: {loss}")

        plt.plot(train_set[:40].numpy(),color = 'red')
        plt.plot(preds, color = 'green')
        
        plt.show()
 

    def validate(self):

        self.window_size = 24*7
        self.future =  24
        train_set,test_set = self.import_process_data("./data/validate.xlsx")
        #test_set,train_set = self.normalize()
        train_data = self.input_data(train_set, self.window_size)


        model = LSTM(input_size = 1, hidden_size = 100, out_size = 1)
        model.load_state_dict(torch.load(self.trained_model_name))
        model.eval()
        criterion = nn.MSELoss()


        preds = train_set[-self.window_size:].tolist()
        for f in tqdm(range(self.future)):
            seq = torch.FloatTensor(preds[-self.window_size:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1,1,model.hidden_size),
                                torch.zeros(1,1,model.hidden_size))
                preds.append(model(seq).item())
                
        #print('pred',preds[self.window_size:])
        #print('target',test_set[-self.window_size:])

        fig = plt.figure(figsize=(12,4))
        plt.title('Prices Predicted')
        plt.ylabel('Price normalized')
        plt.grid(True)
        plt.autoscale(axis='x',tight=True)
        fig.autofmt_xdate()

        plt.plot(test_set[:24*7], color='blue')
        plt.plot(preds, color='red')
        plt.show()


        loss = criterion(torch.tensor(preds[-1:]), test_set[-1:])
        print(loss)

    def predict(self,window_size, future, data):
        self.window_size = window_size
        self.future = future
        model = LSTM(input_size = 1, hidden_size = 100, out_size = 1)
        model.load_state_dict(torch.load(self.trained_model_name))
        model.eval()

        preds = data[-self.window_size:]
        for f in tqdm(range(self.future)):
            seq = torch.FloatTensor(preds[-self.window_size:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1,1,model.hidden_size),
                                torch.zeros(1,1,model.hidden_size))
                preds.append(model(seq).item())
        return preds[-1]      


    



lstm = LSTM_price()
out= lstm.input_data([0.0128, 0.0104, 0.0098, 0.0092, 0.0080],3)
a = lstm.predict(3,1,out[0][0])
#print(out[0][0])
print(a)
print(out[0][1])
lstm.validate()
