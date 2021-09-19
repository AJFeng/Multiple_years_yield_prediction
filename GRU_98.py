# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:12:22 2020

@author: aijing
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import seaborn as sns
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import ipdb
from random import randrange, seed
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datetime import datetime as dt
import time
import pickle
import math
import xgboost as xgb
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import statsmodels.api as sm
from statistics import mean



import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(1234)


if torch.cuda.is_available():
    for gpu_i in range(torch.cuda.device_count()):
        print('Avalible GPU: '+torch.cuda.get_device_name('cuda:'+str(gpu_i)), flush = True)

device = torch.device('cuda:'+str(torch.cuda.device_count()-1) if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name('cuda:'+str(torch.cuda.device_count()-1))+' is used', flush = True)

#--------------------------------- hyperparameters -------------------------------------------------------------------
soil_kernel_1=4
soil_kernel_2=8

weather_kernel_1=8
weather_kernel_2=12

training_years=['2018', '2017','2019']
testing_years=['2019W']

BO_n_iter=100
BO_num=0
chosen_score='MAE' # R2 or MAE
setted_epochs_num=200

# True means this component will be used
S_CNN_flag=True
W_CNN_flag=True
TF_test_flag=True
TF_train_flag=True
S2Y_flag=True
GRU_flag=True
no_GRU_NDVI_flag=True   # under the model without GRU, NDVI still will not be used if False (WO_GRU_NDVI) but will be used if True (WO _GRU). Weather will always be used
no_GRU_NDVI_W_flag=True  # under the model without GRU, NDVI and weather will not be used if False (WO_GRU_NDVI_W). 

kernel_number_test_flag=False
component_test_flag=False

pre_model_parameters_path=''

def kernel_number_and_compoent_flags():
    global kernel_number_test_flag,component_test_flag
    
    if (soil_kernel_1==4) and (soil_kernel_2==8) and (weather_kernel_1==8) and (weather_kernel_2==12):
        kernel_number_test_flag=False
    else:
        kernel_number_test_flag=True
    
    if (S_CNN_flag==True) and (W_CNN_flag==True) and (TF_test_flag==True) and (TF_train_flag==True) and (S2Y_flag==True) and (GRU_flag==True):
        component_test_flag=False
    else:
        component_test_flag=True
    
def print_globals_setting():
    print('-----------------kernel numbers------------------------------------------',flush = True)
    print('soil_kernel_1: ',soil_kernel_1,flush = True)
    print('soil_kernel_2: ',soil_kernel_2,flush = True)
    print('weather_kernel_1: ',weather_kernel_1,flush = True)
    print('weather_kernel_2: ',weather_kernel_2,flush = True)
    
    print('-----------------traning and testing set------------------------------------------',flush = True)
    print('training_years: ',training_years,flush = True)
    print('testing_years: ',testing_years,flush = True)
    print('BO_n_iter: ',BO_n_iter,flush = True)
    print('setted_epochs_num: ',setted_epochs_num,flush = True)
    
    print('-----------------components flags------------------------------------------',flush = True)
    print('S_CNN_flag: ',S_CNN_flag,flush = True)
    print('W_CNN_flag: ',W_CNN_flag,flush = True)
    print('TF_test_flag: ',TF_test_flag,flush = True)
    print('TF_train_flag: ',TF_train_flag,flush = True)
    print('S2Y_flag: ',S2Y_flag,flush = True)
    print('GRU_flag: ',GRU_flag,flush = True)
    print('no_GRU_NDVI_flag: ',no_GRU_NDVI_flag,flush = True)
    print('no_GRU_NDVI_W_flag: ',no_GRU_NDVI_W_flag,flush = True)
    print('kernel_number_test_flag: ',kernel_number_test_flag,flush = True)
    print('component_test_flag: ',component_test_flag,flush = True)

def hyperparameters_default():
    
    global kernel_number_test_flag,component_test_flag
    global soil_kernel_1,soil_kernel_2,weather_kernel_1, weather_kernel_2,training_years,testing_years
    global S_CNN_flag,W_CNN_flag,TF_test_flag,TF_train_flag,S2Y_flag,GRU_flag
    global no_GRU_NDVI_flag,no_GRU_NDVI_W_flag
    global setted_epochs_num
    
    soil_kernel_1=4
    soil_kernel_2=8
    
    weather_kernel_1=8
    weather_kernel_2=12
    
    training_years=['2018', '2017','2019']
    testing_years=['2019W']
    
    BO_n_iter=100
    chosen_score='R2' # R2 or MAE
    setted_epochs_num=200
    
    S_CNN_flag=True
    W_CNN_flag=True
    TF_test_flag=True
    TF_train_flag=True
    S2Y_flag=True
    GRU_flag=True
    no_GRU_NDVI_flag=True   # under the model without GRU, NDVI still will not be used if False (WO_GRU_NDVI) but will be used if True (WO _GRU). Weather will always be used
    no_GRU_NDVI_W_flag=True  # under the model without GRU, NDVI and weather will not be used if False (WO_GRU_NDVI_W). 

    
    kernel_number_test_flag=False
    component_test_flag=False
    
    kernel_number_and_compoent_flags()
    
def hyperparameters_setting(change_names,change_values):
    
    for name, values in zip(change_names,change_values):
        globals()[name]=values
        
    kernel_number_and_compoent_flags()
    
#------------------------------  LSTM model    -------------------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_layers):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size =input_size
        

        self.gru = nn.GRU(input_size, hidden_size=hidden_size,num_layers =n_layers)

    def forward(self, inputs, hidden):
           
        output, hidden = self.gru(inputs, hidden)
        return output, hidden

    
    
class Soil_h0(nn.Module):
    def __init__(self):
        super(Soil_h0, self).__init__()
        self.conv1=nn.Conv1d(in_channels=1, out_channels=soil_kernel_1, kernel_size=2, stride=1)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2,padding=0)
        self.conv2=nn.Conv1d(in_channels=soil_kernel_1, out_channels=soil_kernel_2, kernel_size=2, stride=1)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=1,padding=0)
        self.relu=nn.Tanh()   # relu would result all zero in the ouput
        self.fc=nn.Linear(4,1)

        
    
    def forward(self,soil):
        
        #ipdb.set_trace()
        clay=torch.tensor(soil[:,1:8])
        clay=clay.unsqueeze(1) # size: [2394, 1, 7]
        
        
        soil_other=torch.cat((soil[:,0].unsqueeze(1),soil[:,8:11]),dim=1)
        soil_other=torch.tensor(soil_other)
        soil_other=self.fc(soil_other.float())
        
        out=self.relu(self.conv1(clay.float()))
        out=self.avg1(out)
        out=self.relu(self.conv2(out))
        clay_encoder=self.avg2(out)
        clay_encoder=clay_encoder.squeeze(2)

        return torch.cat((soil_other,clay_encoder),dim=1)

class Weather_inputs(nn.Module):
    def __init__(self):
        super(Weather_inputs, self).__init__()
        self.conv1=nn.Conv1d(in_channels=6, out_channels=weather_kernel_1, kernel_size=3, stride=1)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2,padding=0)
        self.conv2=nn.Conv1d(in_channels=weather_kernel_1, out_channels=weather_kernel_2, kernel_size=3, stride=1)
        self.relu=nn.Tanh()
        
        self.conv3=nn.Conv1d(in_channels=6, out_channels=weather_kernel_1, kernel_size=3, stride=1)
        self.conv4=nn.Conv1d(in_channels=weather_kernel_1, out_channels=weather_kernel_2, kernel_size=1, stride=1)
        
        self.conv5=nn.Conv1d(in_channels=6, out_channels=weather_kernel_1, kernel_size=3, stride=1)
        self.conv6=nn.Conv1d(in_channels=weather_kernel_1, out_channels=weather_kernel_2, kernel_size=1, stride=1)
        
        self.conv7=nn.Conv1d(in_channels=6, out_channels=weather_kernel_1, kernel_size=3, stride=1)
        self.conv8=nn.Conv1d(in_channels=weather_kernel_1, out_channels=weather_kernel_2, kernel_size=1, stride=1)

        
    def forward(self,weather,batch_size):
        
        weather_13=self.W_CNN_13(weather[:,:,0:13].float())  # weather size: [batch_size,6,26], weather_13 size: [batch_size,12]
        #ipdb.set_trace()
        #weather_13=weather_13.expand(batch_size,weather_kernel_2)
        weather_13=weather_13.unsqueeze(0)
        weather_5=self.W_CNN_5(weather[:,:,13:18].float())
        #weather_5=weather_5.expand(batch_size,weather_kernel_2)
        weather_5=weather_5.unsqueeze(0)
        weather_4_1=self.W_CNN_4_1(weather[:,:,18:22].float())
        #weather_4_1=weather_4_1.expand(batch_size,weather_kernel_2)
        weather_4_1=weather_4_1.unsqueeze(0)
        weather_4_2=self.W_CNN_4_2(weather[:,:,22:26].float())
        #weather_4_2=weather_4_2.expand(batch_size,weather_kernel_2)
        weather_4_2=weather_4_2.unsqueeze(0)

        #ipdb.set_trace()
        return torch.cat((weather_13,weather_5,weather_4_1,weather_4_2),dim=0)
    
    def W_CNN_13(self,data):
                
        out=self.relu(self.conv1(data)) # data size: [batch_size,6,13]
        out=self.avg1(out)
        out=self.relu(self.conv2(out))
        out=self.avg1(out)
        
        out=out.squeeze(2)
        
        return out
        
    
    def W_CNN_5(self,data):
                
        out=self.relu(self.conv3(data))
        out=self.avg1(out)
        out=self.relu(self.conv4(out))
        
        out=out.squeeze(2)
        
        return out   
        
    
    def W_CNN_4_1(self,data):
        
        out=self.relu(self.conv5(data))
        out=self.avg1(out)
        out=self.relu(self.conv6(out))
        
        out=out.squeeze(2)
           
        return out
    
    def W_CNN_4_2(self,data):
        
        out=self.relu(self.conv7(data))
        out=self.avg1(out)
        out=self.relu(self.conv8(out))
        
        out=out.squeeze(2)
           
        return out
        
class LSTM_Model(nn.Module):
    def __init__(self, seq_size,batch_size,n_hidden,soil_encode, rnn, weather_encode):
        super().__init__()
        
        self.soil_encode = soil_encode
        self.weather_encode = weather_encode
        self.rnn = rnn
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.fc=nn.Linear(soil_kernel_2+1,n_hidden)
        self.fc_raw_soil=nn.Linear(11,n_hidden)
        self.fc2=nn.Linear(n_hidden,1)
        self.relu=nn.Tanh()
        self.fc_yield1=nn.Linear(n_hidden+3,8)
        self.fc_yield2=nn.Linear(8,4)
        self.fc_yield3=nn.Linear(5,1)
        self.fc_yield3_no_S2Y=nn.Linear(4,1)
        self.fc_soil_final=nn.Linear(soil_kernel_2+1,1)
        self.fc_soil_final_raw_soil=nn.Linear(11,1)
        self.n_hidden=n_hidden
        
    def forward(self, soil,weather1,irrigation_data,crop_plant_date,img_yield_sampling,n_layers,teacher_forcing_ratio=0):
        
        global soil_kernel_1,soil_kernel_2,weather_kernel_1, weather_kernel_2
        global S_CNN_flag,W_CNN_flag,TF_test_flag,TF_train_flag,S2Y_flag,GRU_flag
        
        outputs = torch.zeros(self.seq_size+1, self.batch_size, 1).to(device)
        #ipdb.set_trace()
        #soil.device
        
        if S_CNN_flag==True:
            soil_h0=self.soil_encode(soil) 
            soil_h0=self.relu(self.fc(soil_h0))
        if S_CNN_flag==False:
            soil=torch.tensor(soil)
            soil_h0=self.relu(self.fc_raw_soil(soil[:,0:11].float()))
            
        soil_h0=soil_h0.unsqueeze(0)
        hidden=soil_h0.expand([n_layers,self.batch_size,self.n_hidden])  # size:[1,BZ,HZ]
        
        #
        # weather1 size: [1,6,26], irrigation_plots size: [batch_size,2], irrigation_data size: [5*4, -1], crop_plant_date type: int 1*1
        weather1=weather1.repeat(self.batch_size,1,1) # weather size: [batch_size,6,26], the 'batch_size' dimension is the repeat value
        irrigation_plots=soil[:,11:]
        if irrigation_data.size>40: # if no irrigation data, dimension is 20*2
            for irrigation_date in irrigation_data.columns[2:]:
                week_number=math.ceil((int(irrigation_date)+crop_plant_date)/7)
                for batch_num in range(self.batch_size):
                    #ipdb.set_trace()
                    irrigate_amount=np.array(irrigation_data[irrigation_date][(irrigation_data['irrigation_length']==int(irrigation_plots[batch_num,0])) & (irrigation_data['irrigation_circle']==int(irrigation_plots[batch_num,1]))])
                    weather1[batch_num,0,week_number]=weather1[batch_num,0,week_number]+torch.tensor(irrigate_amount/7).to(device)

        #ipdb.set_trace()
        
        if W_CNN_flag==True:
            weather_temp1=self.weather_encode(weather1,self.batch_size) # size:[4,64,12]
        if W_CNN_flag==False:
            weather_13=torch.mean(weather1[:,:,0:13].float(),2)  # weather1 size: [batch_size,6,26], weather_13 size: [batch_size,6]
            weather_13=weather_13.unsqueeze(0) # weather_13 size: [1,batch_size,6]
            weather_5=torch.mean(weather1[:,:,13:18].float(),2)
            weather_5=weather_5.unsqueeze(0)
            weather_4_1=torch.mean(weather1[:,:,18:22].float(),2)
            weather_4_1=weather_4_1.unsqueeze(0)
            weather_4_2=torch.mean(weather1[:,:,22:26].float(),2)
            weather_4_2=weather_4_2.unsqueeze(0)
            weather_temp1=torch.cat((weather_13,weather_5,weather_4_1,weather_4_2),dim=0) # weather_temp1 size: [4,batch_size,6]


        init=torch.zeros(1, self.batch_size, 1).to(device)
        inputs=torch.cat((weather_temp1[0,:,:].unsqueeze(0),init),dim=2).float() # size: [1, BZ, 13]
        
        img_yield_sampling=torch.tensor(img_yield_sampling) # size:[64,4]
        
        for t in range(len(weather_temp1)):
            
            output, hidden=self.rnn(inputs,hidden) # hidden size:[1,BZ,HZ] output size:[1,BZ,HZ] 
            output=self.relu(self.fc2(output)) # output size:[1,BZ,1]
            outputs[t,:,:] = output
            #ipdb.set_trace()
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio # 0.7
            
            if t==len(weather_temp1)-1:
                output, hidden=self.rnn(inputs,hidden)
                output=self.relu(self.fc2(output))
                outputs[t,:,:] = output
                
                #output=torch.cat((outputs[0:t,:,:].permute(2,1,0),output),dim=2).float()
                #output=self.relu(self.fc_yield2(self.relu(self.fc_yield1(output))))
                
                #xgboost_pred=xgbmodel.predict(np.concatenate((soil.detach().numpy(),outputs[1,:,:].detach().numpy()),axis=1))
                if (S_CNN_flag==True) and (S2Y_flag==True):
                    xgboost_pred=self.relu(self.fc_soil_final(self.soil_encode(soil))).to(device)
                if (S_CNN_flag==False) and (S2Y_flag==True):
                    xgboost_pred=self.relu(self.fc_soil_final_raw_soil(soil[:,0:11].float())).to(device)
                if S2Y_flag==True:
                    output=torch.cat((outputs[0:t+1,:,:],xgboost_pred.unsqueeze(0)),dim=0).float() # size: [5, BZ, 1]
                    output=self.fc_yield3(output.permute(2,1,0)) # size: [1, BZ, 5]
                if S2Y_flag==False:
                    output=self.fc_yield3_no_S2Y(outputs[0:t+1,:,:].permute(2,1,0)) # size: [1, BZ, 5]
                outputs[t+1,:,:] = output
                break
            
            if teacher_force:
                img_temp = img_yield_sampling[:,t].float()
                check_nan=img_temp==-100
                img_temp[check_nan]=output.squeeze(0).squeeze(1)[check_nan]
                img =img_temp.unsqueeze(0).unsqueeze(2).float()  
            else:
                img =output            
            inputs=torch.cat((weather_temp1[t+1,:,:].unsqueeze(0),img),dim=2).float()

            
            
        
        
        return outputs



#-------------------------------------  read soil and weather data -----------------------------------------------------
soil_E=pd.read_csv('soil features downsample.csv')[['sand%','clay10','clay20','clay30','clay40','clay50',
                'clay60','clay70','FC','WP','TAW','irrigation_length','irrigation_circle']]
soil_W=pd.read_csv('soil features downsample W.csv')[['sand%','clay10','clay20','clay30','clay40','clay50',
                'clay60','clay70','FC','WP','TAW','irrigation_length','irrigation_circle']]

scalersoil = StandardScaler()
soil_E.iloc[:,0:11]=scalersoil.fit_transform(soil_E.iloc[:,0:11])
soil_W.iloc[:,0:11]=scalersoil.fit_transform(soil_W.iloc[:,0:11])

irrigation_data_2019W=pd.read_csv('irrigation data 2019W.csv')
irrigation_data_2019=pd.read_csv('irrigation data 2019.csv')
irrigation_data_2018=pd.read_csv('irrigation data 2018.csv')
irrigation_data_2017=pd.read_csv('irrigation data 2017.csv')

plantting_date={'2017':22,'2018':15,'2019':14, '2019W':14}   # use May 1st as the base
 
   
weather_2019=pd.read_csv('weather2019_LSTM.csv').drop(['MONTH','DAY','YEAR'],axis=1) # size: [182,6]
m = nn.AvgPool1d(kernel_size=7, stride=7,padding=0)  # daily weather to weekly weather
temp_weather=torch.tensor(np.transpose(np.array(weather_2019)))  # size: [6,182], the [0,:] is the PRECIP 
temp_weather=temp_weather.unsqueeze(0) # size: [1,6,182]
weather2019W=weather2019=m(temp_weather) # size: [1,6,26]

weather_2017=pd.read_csv('weather2017_LSTM.csv').drop(['MONTH','DAY','YEAR'],axis=1)
m = nn.AvgPool1d(kernel_size=7, stride=7,padding=0)
temp_weather=torch.tensor(np.transpose(np.array(weather_2017)))  # size: [6,182]
temp_weather=temp_weather.unsqueeze(0) # size: [1,6,182]
weather2017=m(temp_weather) # size: [1,6,26]


weather_2018=pd.read_csv('weather2018_LSTM.csv').drop(['MONTH','DAY','YEAR'],axis=1)
m = nn.AvgPool1d(kernel_size=7, stride=7,padding=0)
temp_weather=torch.tensor(np.transpose(np.array(weather_2018)))  # size: [6,182]
temp_weather=temp_weather.unsqueeze(0) # size: [1,6,182]
weather2018=m(temp_weather) # size: [1,6,26]

weather_8years_ave=pd.read_csv('8 year_mean_weather_LSTM.csv') # size: [26,6]
temp_weather=torch.tensor(np.transpose(np.array(weather_8years_ave)))  # size: [6,26]
weather8years=temp_weather.unsqueeze(0) # size: [1,6,26]



img_08_2017=pd.read_csv('data_08_2017 downsample.csv')[['NDVI']]
yield_2017=pd.read_csv('data_08_2017 downsample.csv')[['Yield2017']]


scaler2017 = StandardScaler()
scaler_08_2017 = StandardScaler()
img_08_2017=scaler_08_2017.fit_transform(img_08_2017)
yield_2017=scaler2017.fit_transform(yield_2017)


img_09_2018=pd.read_csv('data_09_2018 downsample.csv')[['NDVI']]
img_08_2018=pd.read_csv('data_08_2018 downsample.csv')[['NDVI']]
img_07_2018=pd.read_csv('data_07_2018 downsample.csv')[['NDVI']]
yield_2018=pd.read_csv('data_09_2018 downsample.csv')[['Yield2018']]
yield_2019W=pd.read_csv('data_09_2018 downsample.csv')[['Yield2019']]

scaler2018 = StandardScaler()
scaler_07_2018 = StandardScaler()
scaler_08_2018 = StandardScaler()
scaler_09_2018 = StandardScaler()
scaler2019W = StandardScaler()
img_07_2018=scaler_07_2018.fit_transform(img_07_2018)
img_08_2018=scaler_08_2018.fit_transform(img_08_2018)
img_09_2018=scaler_09_2018.fit_transform(img_09_2018)
yield_2018=scaler2018.fit_transform(yield_2018)
yield_2019W=scaler2019W.fit_transform(yield_2019W)


img_07_2019=pd.read_csv('data_07 downsample.csv')[['NDVI']]
img_08_2019=pd.read_csv('data_08 downsample.csv')[['NDVI']]
img_09_2019=pd.read_csv('data_09 downsample.csv')[['NDVI']]
yield_2019=pd.read_csv('data_08 downsample.csv')[['Yield2019']]


row_col_E=pd.read_csv('soil features downsample.csv')[['row','col']]
row_col_W=pd.read_csv('soil features downsample W.csv')[['row','col']]
row_col_E_120=pd.read_csv('soil features downsample - 09.csv')[['row','col']]


index=np.where((img_07_2019['NDVI']>0.6) &
                   (img_08_2019['NDVI']>0.6) &
                    (yield_2019['Yield2019']>0.6))

scaler2019 = StandardScaler()
scaler_07_2019 = StandardScaler()
scaler_08_2019 = StandardScaler()
scaler_09_2019 = StandardScaler()
img_07_2019=scaler_07_2019.fit_transform(img_07_2019)
img_08_2019=scaler_08_2019.fit_transform(img_08_2019)
img_09_2019=scaler_09_2019.fit_transform(img_09_2019)
yield_2019=scaler2019.fit_transform(yield_2019)

train_index=row_col_E['row'] <31   # 2019E.09 only have 30 rows for the image data due to camera problems
test_index=row_col_E['row'] >30
yield_500=np.resize(scaler2019.inverse_transform(yield_2019),(len(yield_2019),))>500
yield_300=np.resize(scaler2019.inverse_transform(yield_2019[train_index]),(len(yield_2019[train_index]),))>500
img_yield=pd.DataFrame(np.concatenate([soil_E[train_index & yield_500],img_07_2019[train_index & yield_500],img_08_2019[train_index & yield_500],img_09_2019[yield_300],yield_2019[train_index & yield_500]],axis=1),index=soil_E[train_index & yield_500].index).dropna()
a = np.empty((len(img_07_2019[test_index & yield_500]),1))
a[:] = -100
img_yield_test=pd.DataFrame(np.concatenate([soil_E[test_index & yield_500],img_07_2019[test_index & yield_500],img_08_2019[test_index & yield_500],a,yield_2019[test_index & yield_500]],axis=1),index=soil_E[test_index & yield_500].index).dropna()
img_yield_2019=pd.concat([img_yield,img_yield_test],axis=0)


yield_500=np.resize(scaler2017.inverse_transform(yield_2017),(len(yield_2017),))>500
a = np.empty((len(img_08_2017[yield_500]),1))
a[:] = -100
img_yield_2017=pd.DataFrame(np.concatenate([soil_E[yield_500],a,img_08_2017[yield_500],a,yield_2017[yield_500]],axis=1),index=soil_E[yield_500].index).dropna()


yield_300=np.resize(scaler2018.inverse_transform(yield_2018),(len(yield_2018),))>500
img_yield_2018=pd.DataFrame(np.concatenate([soil_W[yield_300],img_07_2018[yield_300],img_08_2018[yield_300],img_09_2018[yield_300],yield_2018[yield_300]],axis=1),index=soil_W[yield_300].index).dropna()

yield_300=np.resize(scaler2019W.inverse_transform(yield_2019W),(len(yield_2019W),))>500
a = np.empty((len(yield_2019W[yield_300]),1))
a[:] = -100
img_yield_2019W=pd.DataFrame(np.concatenate([soil_W[yield_300],a,a,a,yield_2019W[yield_300]],axis=1),index=soil_W[yield_300].index).dropna()


class soil_img_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, soil, img_yield):
        
        self.soil = soil
        self.img_yield = img_yield

    def __len__(self):
        return len(self.soil)

    def __getitem__(self, idx):
        
        temp_img_yield=self.img_yield[idx].to(device)
        temp_soil=self.soil[idx].to(device)

        return temp_soil,temp_img_yield

# 2019 of 07,08,09 only left part of the field, 85%+15% train and test
#train_index,test_index=train_test_split(index[0], test_size=0.15, random_state=39)    
#dataset_train=soil_img_Dataset(torch.tensor(np.array(soil_E_120.iloc[train_index])),torch.tensor(np.array(img_yield.iloc[train_index])))
#dataset_test=soil_img_Dataset(torch.tensor(np.array(soil_E_120.iloc[test_index])),torch.tensor(np.array(img_yield.iloc[test_index])))

# 2019 of 07,08,09, use left part of the field to predict the right part of the field
#dataset_train=soil_img_Dataset(torch.tensor(np.array(img_yield.iloc[:,0:11])),torch.tensor(np.array(img_yield.iloc[:,11:])))
#dataset_test=soil_img_Dataset(torch.tensor(np.array(img_yield_test.iloc[:,0:11])),torch.tensor(np.array(img_yield_test.iloc[:,11:])))

torch.manual_seed(1234)

dataset_train1=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+training_years[0]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+training_years[0]].iloc[:,13:])))
dataset_train2=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+training_years[1]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+training_years[1]].iloc[:,13:])))
if len(training_years)>2:
    dataset_train3=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+training_years[2]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+training_years[2]].iloc[:,13:])))

dataset_test=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+testing_years[0]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+testing_years[0]].iloc[:,13:])))
img_yield_test_reversed=globals()['img_yield_'+testing_years[0]][::-1]
dataset_test2=soil_img_Dataset(torch.tensor(np.array(img_yield_test_reversed.iloc[:,0:13])),torch.tensor(np.array(img_yield_test_reversed.iloc[:,13:])))




#------------------------------------------ XGBoost  --------------------------------------------------------------

def xgb_r2_score(preds, dtrain):
    # Courtesy of Tilii
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

def train_xgb(max_depth, subsample,C, reg_lambda):
    # Evaluate an XGBoost model using given params
    xgb_params = {
        'n_trees': 250,
        'eta': 0.01,
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'objective': 'reg:squarederror',
        'silent': 1,  #0: print the processing result, 1: not print 
        'C': max(C, 0),
        'reg_lambda': max(reg_lambda,0)
    }
    scores = xgb.cv(xgb_params, dtrain, num_boost_round=1500, early_stopping_rounds=50, verbose_eval=False, feval=xgb_r2_score, maximize=True, nfold=5,seed=1234)['test-r2-mean'].iloc[-1]
    return scores

params_xgb = {
  'C':(10e-2, 1),
  'subsample':(0.6, 0.8),
  'max_depth': (3, 7),
  'reg_lambda':(0.2,0.8)
}

dtrain =xgb.DMatrix([])

def xgboost_model():
    
    global weather2017,weather2018,weather2019,soil_E,soil_W,img_08_2019,img_08_2018,img_08_2017,yield_2019W,yield_2019,yield_2018,yield_2017
    global testing_years,GRU_flag,no_GRU_NDVI_flag,no_GRU_NDVI_W_flag
    
    if GRU_flag==True:
        print('The GRU_flag is True!')
        return
    
    weather1=weather2017
    weather_13=torch.mean(weather1[:,:,0:13].float(),2)
    weather_13=weather_13.expand(1,6)
    weather_5=torch.mean(weather1[:,:,13:18].float(),2)
    weather_5=weather_5.expand(1,6)
    weather_4_1=torch.mean(weather1[:,:,18:22].float(),2)
    weather_4_1=weather_4_1.expand(1,6)
    weather_4_2=torch.mean(weather1[:,:,22:26].float(),2)
    weather_4_2=weather_4_2.expand(1,6)
    
    weather_temp1=torch.cat((weather_13,weather_5,weather_4_1,weather_4_2),dim=1)
    weather_temp1=weather_temp1.detach().numpy()   # size=[1,6*4]
    weather_temp2017=weather_temp1
    
    weather1=weather2018
    weather_13=torch.mean(weather1[:,:,0:13].float(),2)
    weather_13=weather_13.expand(1,6)
    weather_5=torch.mean(weather1[:,:,13:18].float(),2)
    weather_5=weather_5.expand(1,6)
    weather_4_1=torch.mean(weather1[:,:,18:22].float(),2)
    weather_4_1=weather_4_1.expand(1,6)
    weather_4_2=torch.mean(weather1[:,:,22:26].float(),2)
    weather_4_2=weather_4_2.expand(1,6)
    
    weather_temp1=torch.cat((weather_13,weather_5,weather_4_1,weather_4_2),dim=1)
    weather_temp1=weather_temp1.detach().numpy()   # size=[1,6*4]
    weather_temp2018=weather_temp1
    
    weather1=weather2019
    weather_13=torch.mean(weather1[:,:,0:13].float(),2)
    weather_13=weather_13.expand(1,6)
    weather_5=torch.mean(weather1[:,:,13:18].float(),2)
    weather_5=weather_5.expand(1,6)
    weather_4_1=torch.mean(weather1[:,:,18:22].float(),2)
    weather_4_1=weather_4_1.expand(1,6)
    weather_4_2=torch.mean(weather1[:,:,22:26].float(),2)
    weather_4_2=weather_4_2.expand(1,6)
    
    weather_temp1=torch.cat((weather_13,weather_5,weather_4_1,weather_4_2),dim=1)
    weather_temp1=weather_temp1.detach().numpy()   # size=[1,6*4]
    weather_temp2019=weather_temp1
    
    if (testing_years[0]=='2019') and (no_GRU_NDVI_flag==True):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E),axis=0)),pd.DataFrame(np.concatenate((np.tile(weather_temp2018,(2331,1)),np.tile(weather_temp2017,(2394,1))),axis=0)),pd.DataFrame(np.concatenate((img_08_2018,img_08_2017),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2017),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2019
    if (testing_years[0]=='2019') and (no_GRU_NDVI_flag==False) and (no_GRU_NDVI_W_flag==True):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E),axis=0)),pd.DataFrame(np.concatenate((np.tile(weather_temp2018,(2331,1)),np.tile(weather_temp2017,(2394,1))),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2017),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2019
    if (testing_years[0]=='2019') and (no_GRU_NDVI_W_flag==False):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2017),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2019
       
    if (testing_years[0]=='2018') and (no_GRU_NDVI_flag==True):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_E,soil_E),axis=0)),pd.DataFrame(np.concatenate((np.tile(weather_temp2019,(2394,1)),np.tile(weather_temp2017,(2394,1))),axis=0)),pd.DataFrame(np.concatenate((img_08_2019,img_08_2017),axis=0)),pd.DataFrame(np.concatenate((yield_2019,yield_2017),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2018
    if (testing_years[0]=='2018') and (no_GRU_NDVI_flag==False) and (no_GRU_NDVI_W_flag==True):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_E,soil_E),axis=0)),pd.DataFrame(np.concatenate((np.tile(weather_temp2019,(2394,1)),np.tile(weather_temp2017,(2394,1))),axis=0)),pd.DataFrame(np.concatenate((yield_2019,yield_2017),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2018
    if (testing_years[0]=='2018') and (no_GRU_NDVI_W_flag==False):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_E,soil_E),axis=0)),pd.DataFrame(np.concatenate((yield_2019,yield_2017),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2018
        
    if (testing_years[0]=='2017') and (no_GRU_NDVI_flag==True):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E),axis=0)),pd.DataFrame(np.concatenate((np.tile(weather_temp2018,(2331,1)),np.tile(weather_temp2019,(2394,1))),axis=0)),pd.DataFrame(np.concatenate((img_08_2018,img_08_2019),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2019),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2017
    if (testing_years[0]=='2017') and (no_GRU_NDVI_flag==False) and (no_GRU_NDVI_W_flag==True):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E),axis=0)),pd.DataFrame(np.concatenate((np.tile(weather_temp2018,(2331,1)),np.tile(weather_temp2019,(2394,1))),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2019),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2017
    if (testing_years[0]=='2017') and (no_GRU_NDVI_W_flag==False):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2019),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2017
        
    if (testing_years[0]=='2019W') and (no_GRU_NDVI_W_flag==True):  #no NDVI data for 2019W
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E,soil_E),axis=0)),pd.DataFrame(np.concatenate((np.tile(weather_temp2018,(2331,1)),np.tile(weather_temp2017,(2394,1)),np.tile(weather_temp2019,(2394,1))),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2017,yield_2019),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2019W
    if (testing_years[0]=='2019W') and (no_GRU_NDVI_W_flag==False):
        x_temp=pd.concat([pd.DataFrame(np.concatenate((soil_W,soil_E,soil_E),axis=0)),pd.DataFrame(np.concatenate((yield_2018,yield_2017,yield_2019),axis=0))],axis=1).dropna()
        scaler_xgb=scaler2019W
    
    x_temp=np.array(x_temp)
    global dtrain
    dtrain = xgb.DMatrix(x_temp[:,0:-1], x_temp[:,-1])
    
    xgb_bayesopt = BayesianOptimization(train_xgb, params_xgb)
    
    # Maximize R2 score
    xgb_bayesopt.maximize(init_points=5, n_iter=30)
    
    # Get the best params
    p = xgb_bayesopt.max['params']
    
    xgb_params = {
        'n_trees': 250,
        'eta': 0.01,
        'max_depth': int(p['max_depth']),
        'subsample': max(min(p['subsample'], 1), 0),
        'objective': 'reg:squarederror',
        'silent': 1,
        'C': max(p['C'],0),
        'reg_lambda':max(p['reg_lambda'],0)
    }
    
    xgbmodel_final = xgb.XGBRegressor(**xgb_params,feval=xgb_r2_score, maximize=True,random_state=1234)
    xgbmodel_final.fit(x_temp[:,0:-1], x_temp[:,-1])
    
    if (testing_years[0]=='2019') and (no_GRU_NDVI_flag==True):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_E),pd.DataFrame(np.tile(weather_temp2019,(2394,1))),pd.DataFrame(img_08_2019),pd.DataFrame(yield_2019)],axis=1).dropna())
        predictions_xgb = scaler2019.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2018') and (no_GRU_NDVI_flag==True):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_W),pd.DataFrame(np.tile(weather_temp2018,(2331,1))),pd.DataFrame(img_08_2018),pd.DataFrame(yield_2018)],axis=1).dropna())
        predictions_xgb = scaler2018.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2017') and (no_GRU_NDVI_flag==True):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_E),pd.DataFrame(np.tile(weather_temp2017,(2394,1))),pd.DataFrame(img_08_2017),pd.DataFrame(yield_2017)],axis=1).dropna())
        predictions_xgb = scaler2017.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))

    if (testing_years[0]=='2019') and (no_GRU_NDVI_flag==False) and (no_GRU_NDVI_W_flag==True):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_E),pd.DataFrame(np.tile(weather_temp2019,(2394,1))),pd.DataFrame(yield_2019)],axis=1).dropna())
        predictions_xgb = scaler2019.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2018') and (no_GRU_NDVI_flag==False) and (no_GRU_NDVI_W_flag==True):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_W),pd.DataFrame(np.tile(weather_temp2018,(2331,1))),pd.DataFrame(yield_2018)],axis=1).dropna())
        predictions_xgb = scaler2018.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2017') and (no_GRU_NDVI_flag==False) and (no_GRU_NDVI_W_flag==True):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_E),pd.DataFrame(np.tile(weather_temp2017,(2394,1))),pd.DataFrame(yield_2017)],axis=1).dropna())
        predictions_xgb = scaler2017.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2019W') and (no_GRU_NDVI_W_flag==True):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_W),pd.DataFrame(np.tile(weather_temp2019,(2394,1))),pd.DataFrame(yield_2019W)],axis=1).dropna())
        predictions_xgb = scaler2019W.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))

    if (testing_years[0]=='2019') and (no_GRU_NDVI_W_flag==False):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_E),pd.DataFrame(yield_2019)],axis=1).dropna())
        predictions_xgb = scaler2019.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2018') and (no_GRU_NDVI_W_flag==False):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_W),pd.DataFrame(yield_2018)],axis=1).dropna())
        predictions_xgb = scaler2018.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2017') and (no_GRU_NDVI_W_flag==False):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_E),pd.DataFrame(yield_2017)],axis=1).dropna())
        predictions_xgb = scaler2017.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))
    if (testing_years[0]=='2019W') and (no_GRU_NDVI_W_flag==False):
        x_temp=np.array(pd.concat([pd.DataFrame(soil_W),pd.DataFrame(yield_2019W)],axis=1).dropna())
        predictions_xgb = scaler2019W.inverse_transform(xgbmodel_final.predict(x_temp[:,0:-1]))

    global best_globel_yield_temp,best_globel_output_temp,best_globel_yield_temp2,best_globel_output_temp2
    best_globel_yield_temp=best_globel_yield_temp2=scaler_xgb.inverse_transform(x_temp[:,-1])
    best_globel_output_temp=best_globel_output_temp2=predictions_xgb
    

#---------------------------- train and test the GRU model --------------------------------------------------------------------
def train_LSTM(dataloader_train,weather1_ori,irrigation_data,crop_plant_date, model,optimizer, criterion,batch_size,n_layers,teacher_forcing_ratio):
    
    model.train()
    epoch_loss = 0
    
    model.to(device)
    
    
    #soil_sampling,img_yield_sampling=next(iter(dataloader_train))
    for soil_sampling,img_yield_sampling in dataloader_train:
        
        
        if soil_sampling.size()!=torch.Size([batch_size,13]):
            break
        
        weather1=weather1_ori.clone()
        weather1=weather1.to(device)
        
        model.zero_grad()
        optimizer.zero_grad()
        #xgboost_pred=xgbmodel.predict(soil_sampling.detach().numpy())
        
        # add the irrigation to the TOTAL PRECIP
        
        weather1=weather1.to(device)
        
        if TF_train_flag==True:
            output = model(soil_sampling, weather1,irrigation_data,crop_plant_date,img_yield_sampling,n_layers,teacher_forcing_ratio=teacher_forcing_ratio) # size: [4,64,1]
        else:
            output = model(soil_sampling, weather1,irrigation_data,crop_plant_date,img_yield_sampling,n_layers,teacher_forcing_ratio=0) # size: [4,64,1]

    
        output=output.squeeze(2) # size: [4,64]
        output=output.permute(1,0)
        
        
        check_nan_2=img_yield_sampling[:,2]==-100
        check_nan_1=img_yield_sampling[:,1]==-100
        check_nan_0=img_yield_sampling[:,0]==-100

        yield_loss=criterion(output[:,-1],img_yield_sampling[:,-1].float())
        aug_loss=criterion(output[~check_nan_1,1], torch.tensor(img_yield_sampling[~check_nan_1,1]).float())
        sep_loss = criterion(output[~check_nan_2,2], torch.tensor(img_yield_sampling[~check_nan_2,2]).float())
        jul_loss = criterion(output[~check_nan_0,0], torch.tensor(img_yield_sampling[~check_nan_0,0]).float())
        
        if ~torch.isnan(sep_loss) and ~torch.isnan(jul_loss) and ~torch.isnan(aug_loss):
            loss=yield_loss+aug_loss+sep_loss+jul_loss
        elif ~torch.isnan(sep_loss) and torch.isnan(jul_loss) and ~torch.isnan(aug_loss):
            loss=yield_loss+aug_loss+sep_loss
        elif torch.isnan(sep_loss) and ~torch.isnan(jul_loss) and ~torch.isnan(aug_loss):
            loss=yield_loss+aug_loss+jul_loss
        elif torch.isnan(sep_loss) and torch.isnan(jul_loss) and ~torch.isnan(aug_loss):
            loss=yield_loss+aug_loss
        elif torch.isnan(sep_loss) and torch.isnan(jul_loss) and torch.isnan(aug_loss):
            loss=yield_loss
        
        #yield_loss=np.mean((scaler.inverse_transform(output[:,3])-scaler.inverse_transform(img_yield_sampling[:,3]))/scaler.inverse_transform(img_yield_sampling[:,3]))
        #yield_loss=criterion(output[:,-1],img_yield_sampling[:,-1].float())
        
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()

   
    return epoch_loss / (len(dataloader_train)-1)

def evaluate(dataloader_test,weather1_ori,irrigation_data, crop_plant_date,model,optimizer, criterion,batch_size,n_layers,scaler):
    
    model.to(device)
    model.eval()
    
    epoch_loss = 0
    yield_epoch_loss=0
    output_temp=np.array([])
    yield_temp=np.array([])
    outputAll_temp=np.array([[0,0,0,0,0]])

    
    with torch.no_grad():
       
        for soil_sampling,img_yield_sampling in dataloader_test:
            
            
            
            
            if soil_sampling.size()!=torch.Size([batch_size,13]):
                break
            
            weather1=weather1_ori.clone()
            weather1=weather1.to(device)
            
            
            
            #xgboost_pred=xgbmodel.predict(soil_sampling.detach().numpy())
            if (TF_test_flag==True) and (TF_train_flag==True):
                output = model(soil_sampling, weather1,irrigation_data,crop_plant_date,img_yield_sampling,n_layers,teacher_forcing_ratio=1)
            else:
                output = model(soil_sampling, weather1,irrigation_data,crop_plant_date,img_yield_sampling,n_layers,teacher_forcing_ratio=0)
            # for name, param in model.named_parameters():
            #     print(name, param.data)
            
            output=output.squeeze(2) # size: [5,BZ]
            output=output.permute(1,0) # size: [BZ,5]
            
            
            #loss = criterion(output, torch.tensor(np.array(img_yield_sampling)).float())
            #yield_loss=np.mean(np.abs(scaler.inverse_transform(output[:,-1])-scaler.inverse_transform(img_yield_sampling[:,-1]))/scaler.inverse_transform(img_yield_sampling[:,-1]))
            yield_loss=np.mean(np.abs(scaler.inverse_transform(output[:,-1].cpu().numpy())-scaler.inverse_transform(img_yield_sampling[:,-1].cpu().numpy())))
            output_temp=np.append(output_temp,scaler.inverse_transform(output[:,-1].cpu().numpy()),axis=0)
            yield_temp=np.append(yield_temp,scaler.inverse_transform(img_yield_sampling[:,-1].cpu().numpy()),axis=0)
            
            outputAll_temp=np.append(outputAll_temp,output.cpu().numpy(),axis=0)
            #output_temp=np.append(output_temp,output[:,-1],axis=0)
            #yield_temp=np.append(yield_temp,img_yield_sampling[:,-1],axis=0)
     
            epoch_loss += yield_loss.item()
            yield_epoch_loss += yield_loss.item()
    

    #return epoch_loss / (len(dataloader_test)-1),yield_epoch_loss / (len(dataloader_test)-1), output_temp,yield_temp,outputAll_temp
    return epoch_loss / (len(dataloader_test)),yield_epoch_loss / (len(dataloader_test)), output_temp,yield_temp,outputAll_temp


#------------------------------- BO for GRU model training ---------------------------------------------------------------------
params = {
  'lr':(0.001, 0.1),
  'momentum':(0.8, 1),
  'batch_size': (16, 64),
  'teacher_forcing_ratio':(0,1)
}

best_globel_yield_loss=float('inf')
best_globel_yield_temp=[]
best_globel_output_temp=[]
best_globel_outputAll_temp=[]
best_globel_yield_temp2=[]
best_globel_output_temp2=[]
best_globel_outputAll_temp2=[]
best_globel_R2=0.01

def best_globel__default():
	global best_globel_yield_loss,best_globel_yield_temp,best_globel_output_temp,best_globel_outputAll_temp,best_globel_yield_temp2,best_globel_output_temp2,best_globel_outputAll_temp2,best_globel_R2
    
	best_globel_yield_loss=float('inf')
	best_globel_yield_temp=[]
	best_globel_output_temp=[]
	best_globel_outputAll_temp=[]
	best_globel_yield_temp2=[]
	best_globel_output_temp2=[]
	best_globel_outputAll_temp2=[]
	best_globel_R2=0.01

def train_BO(lr, momentum,batch_size, teacher_forcing_ratio):
    # Evaluate an XGBoost model using given params
    BO_params = {
        'lr': lr,
        'momentum': momentum,
        'batch_size': int(batch_size),
        'teacher_forcing_ratio': teacher_forcing_ratio
        }
    global BO_num
    BO_num=BO_num+1
    R2,yield_loss,_,_,_,_,_,_ = get_scores(BO_params,test_flag=False, save_flag=True)
    
    if chosen_score=='R2':
        score=R2
    elif chosen_score=='MAE':
        score=yield_loss
    
    return score 

def dataset_building(batch_size):
    
    global dataset_train1,dataset_train2,dataset_train3,dataset_test,dataset_test2,training_years,testing_years,img_yield_test_reversed
    torch.manual_seed(1234)

    dataset_train1=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+training_years[0]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+training_years[0]].iloc[:,13:])))
    dataset_train2=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+training_years[1]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+training_years[1]].iloc[:,13:])))
    if len(training_years)>2:
        dataset_train3=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+training_years[2]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+training_years[2]].iloc[:,13:])))
    
    dataset_test=soil_img_Dataset(torch.tensor(np.array(globals()['img_yield_'+testing_years[0]].iloc[:,0:13])),torch.tensor(np.array(globals()['img_yield_'+testing_years[0]].iloc[:,13:])))
    #img_yield_test_reversed=globals()['img_yield_'+testing_years[0]][(len(globals()['img_yield_'+testing_years[0]])//batch_size)*batch_size:-1]
    #img_yield_test_reversed=img_yield_test_reversed.append(pd.DataFrame(np.ones((batch_size-len(img_yield_test_reversed),17))), ignore_index=True)
    img_yield_test_reversed=globals()['img_yield_'+testing_years[0]][::-1]
    dataset_test2=soil_img_Dataset(torch.tensor(np.array(img_yield_test_reversed.iloc[:,0:13])),torch.tensor(np.array(img_yield_test_reversed.iloc[:,13:])))


def get_scores(BO_params,test_flag=True,save_flag=False):
    
    best_valid_loss = float('inf')
    best_yield_loss=float('inf')
    best_yield_temp=[]
    best_output_temp=[]
    best_outputAll_temp=[]
    best_yield_temp2=[]
    best_output_temp2=[]
    best_outputAll_temp2=[]
    best_R2=0.01
    
    global best_globel_yield_loss,best_globel_yield_temp,best_globel_output_temp,best_globel_outputAll_temp,best_globel_yield_temp2,best_globel_output_temp2,best_globel_outputAll_temp2,best_globel_R2
    global setted_epochs_num,BO_num
    
    batch_size=BO_params['batch_size']
    
    n_layers=1
    n_hidden = 64
    if W_CNN_flag==True:
        n_input=weather_kernel_2+1
    if W_CNN_flag==False:
        n_input=6+1
    n_output=1
    seq_size=4
    
    rnn = RNN(n_input, n_hidden, n_output,n_layers)
    
    soil_encode=Soil_h0()
    
    weather_encode=Weather_inputs()

    global dataset_train1,dataset_train2,dataset_train3,dataset_test,dataset_test2
    dataset_building(batch_size)
    
    torch.manual_seed(1234)
    dataloader_train1 = DataLoader(dataset_train1, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    dataloader_train2 = DataLoader(dataset_train2, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    if len(training_years)>2:
        dataloader_train3 = DataLoader(dataset_train3, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    dataloader_test2 = DataLoader(dataset_test2, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    
    model = LSTM_Model(seq_size,batch_size,n_hidden,soil_encode, rnn, weather_encode)
    #optimizer = optim.Adam(model.parameters(),lr=0.005)
    optimizer= optim.SGD(model.parameters(), lr=BO_params['lr'], momentum=BO_params['momentum'])
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    global pre_model_parameters_path
    if test_flag==True:
        model.load_state_dict(torch.load(pre_model_parameters_path,'cuda:'+str(gpu_i)))
        model.eval()
        N_EPOCHS = 1
    else:
        N_EPOCHS = setted_epochs_num
        
    start = time.time()

    for epoch in range(N_EPOCHS):
        
        if test_flag==False:
            train_loss1 = train_LSTM(dataloader_train1,globals()['weather'+training_years[0]],
                                     globals()['irrigation_data_'+training_years[0]],plantting_date[training_years[0]],
                                     model, optimizer, criterion,batch_size,n_layers,BO_params['teacher_forcing_ratio'])
            train_loss2 = train_LSTM(dataloader_train2,globals()['weather'+training_years[1]],
                                     globals()['irrigation_data_'+training_years[1]],plantting_date[training_years[1]],
                                     model, optimizer, criterion,batch_size,n_layers,BO_params['teacher_forcing_ratio'])
            if len(training_years)>2:
                train_loss3 = train_LSTM(dataloader_train3,globals()['weather'+training_years[2]],
                                         globals()['irrigation_data_'+training_years[2]],plantting_date[training_years[0]],
                                         model, optimizer, criterion,batch_size,n_layers,BO_params['teacher_forcing_ratio'])
            if len(training_years)>2:
                train_loss=(train_loss1+train_loss2+train_loss3)/3
            else:
                train_loss=(train_loss1+train_loss2)/2
         

        
        valid_loss,yield_loss,output_temp,yield_temp,outputAll_temp = evaluate(dataloader_test, globals()['weather'+testing_years[0]],
                                                                               globals()['irrigation_data_'+testing_years[0]],plantting_date[testing_years[0]],
                                                                               model,optimizer, criterion,batch_size,n_layers,globals()['scaler'+testing_years[0]])
        valid_loss2,yield_loss2,output_temp2,yield_temp2,outputAll_temp2 = evaluate(dataloader_test2, globals()['weather'+testing_years[0]],
                                                                                    globals()['irrigation_data_'+testing_years[0]],plantting_date[testing_years[0]],
                                                                                    model,optimizer, criterion,batch_size,n_layers,globals()['scaler'+testing_years[0]])
        yield_loss=(yield_loss+yield_loss2)/2
        
        
        z = np.polyfit(output_temp, yield_temp, 1)
        p = np.poly1d(z)
        yhat = p(output_temp)
        ybar = sum(yield_temp)/len(yield_temp)
        SST = sum((yield_temp - ybar)**2)
        SSreg = sum((yhat - ybar)**2)
        R2 = SSreg/SST

        
        if (yield_loss+(250/R2)) < (best_yield_loss+(250/best_R2)):
            best_yield_loss = yield_loss
            best_output_temp=output_temp
            best_yield_temp=yield_temp
            best_outputAll_temp=outputAll_temp
            best_output_temp2=output_temp2
            best_yield_temp2=yield_temp2
            best_outputAll_temp2=outputAll_temp2
            best_R2=R2
            #torch.save(model.state_dict(), 'lstm-model_2019W.pt')
            
        if (yield_loss+(250/R2)) < (best_globel_yield_loss+(250/best_globel_R2)):
            best_globel_yield_loss = yield_loss
            best_globel_output_temp=output_temp
            best_globel_yield_temp=yield_temp
            best_globel_outputAll_temp=outputAll_temp
            best_globel_output_temp2=np.flip(output_temp2)
            best_globel_yield_temp2=np.flip(yield_temp2)
            best_globel_outputAll_temp2=np.flip(outputAll_temp2)
            best_globel_R2=R2
            # ts = time.time()
            # st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            
            if save_flag==True:
                torch.save(model, str(testing_years[0])+'Skernels'+str(soil_kernel_1)+'-'+str(soil_kernel_2)+
                  '_Wkernels'+str(weather_kernel_1)+'-'+str(weather_kernel_2)+'S'+str(int(S_CNN_flag))+'W'+
                  str(int(W_CNN_flag))+'TF'+str(int(TF_test_flag))+str(int(TF_train_flag))+
                  'S2Y'+str(int(S2Y_flag))+'GRU'+str(int(GRU_flag))+str(int(no_GRU_NDVI_flag))+str(int(no_GRU_NDVI_W_flag))+'.pth')
                torch.save(model.state_dict(), str(testing_years[0])+'Skernels'+str(soil_kernel_1)+'-'+str(soil_kernel_2)+
                  '_Wkernels'+str(weather_kernel_1)+'-'+str(weather_kernel_2)+'S'+str(int(S_CNN_flag))+'W'+
                  str(int(W_CNN_flag))+'TF'+str(int(TF_test_flag))+str(int(TF_train_flag))+
                  'S2Y'+str(int(S2Y_flag))+'GRU'+str(int(GRU_flag))+str(int(no_GRU_NDVI_flag))+str(int(no_GRU_NDVI_W_flag))+'.pt')
                f = open(str(testing_years[0])+'Skernels'+str(soil_kernel_1)+'-'+str(soil_kernel_2)+
                  '_Wkernels'+str(weather_kernel_1)+'-'+str(weather_kernel_2)+'S'+str(int(S_CNN_flag))+'W'+
                  str(int(W_CNN_flag))+'TF'+str(int(TF_test_flag))+str(int(TF_train_flag))+
                  'S2Y'+str(int(S2Y_flag))+'GRU'+str(int(GRU_flag))+str(int(no_GRU_NDVI_flag))+str(int(no_GRU_NDVI_W_flag))+'.pkl',"wb")
                pickle.dump(BO_params,f)
                f.close()
                
                dict_temp={}
                for key in ['best_globel_yield_loss','best_globel_yield_temp','best_globel_output_temp', 'best_globel_outputAll_temp', 'best_globel_yield_temp2', 'best_globel_output_temp2', 'best_globel_outputAll_temp2']:
                    if (str(type(globals()[key]))!= "<class 'module'>") and (str(type(globals()[key]))!= "<class 'function'>")and (str(type(globals()[key]))!= "<class 'dict'>"):
                        dict_temp[key]=globals()[key]
                        
                with open('dict_'+str(testing_years[0])+'Skernels'+str(soil_kernel_1)+'-'+str(soil_kernel_2)+
                          '_Wkernels'+str(weather_kernel_1)+'-'+str(weather_kernel_2)+'S'+str(int(S_CNN_flag))+'W'+
                          str(int(W_CNN_flag))+'TF'+str(int(TF_test_flag))+str(int(TF_train_flag))+
                          'S2Y'+str(int(S2Y_flag))+'GRU'+str(int(GRU_flag))+str(int(no_GRU_NDVI_flag))+str(int(no_GRU_NDVI_W_flag))+'.pickle',"wb") as f:
                    pickle.dump(dict_temp, f)
                
                plt.scatter(best_globel_output_temp, best_globel_yield_temp, c="b", alpha=0.5)
                plt.yticks([0,1000,2000,3000,4000,5000])
                plt.xticks([0,1000,2000,3000,4000,5000])
                plt.savefig(str(testing_years[0])+'Skernels'+str(soil_kernel_1)+'-'+str(soil_kernel_2)+
                          '_Wkernels'+str(weather_kernel_1)+'-'+str(weather_kernel_2)+'S'+str(int(S_CNN_flag))+'W'+
                          str(int(W_CNN_flag))+'TF'+str(int(TF_test_flag))+str(int(TF_train_flag))+
                          'S2Y'+str(int(S2Y_flag))+'GRU'+str(int(GRU_flag))+str(int(no_GRU_NDVI_flag))+str(int(no_GRU_NDVI_W_flag))+'.png')
                plt.close()

            print('file save: epoch-',epoch,'; BO_num-',BO_num)
            print('MAE: ',round(best_globel_yield_loss,0), ', R2: ',round(best_globel_R2,2), flush = True)

            
        
        
        #print(f'Epoch: {epoch+1:02}')
        #print(f'\tTrain Loss: {train_loss:.3f}')
        #print(f'\t Val. Loss: {valid_loss:.3f} | Yield. Loss: {yield_loss:.3f}')
        #print(f'\t Best Val. Loss: {best_valid_loss:.3f} | Best Yield. Loss: {best_yield_loss:.3f}')

    end = time.time()
    print(str(round((end - start)/60,1))+' Now: '+dt.now().strftime("%m/%d/%Y,%H:%M:%S"), flush = True)

    
    return best_R2,-best_yield_loss,best_yield_temp,best_output_temp,best_outputAll_temp,best_output_temp2,best_yield_temp2,best_outputAll_temp2

def print_accuracy():
    
    #-------------------------MAE, MAE% and R2 -----------------------------------------------
    global best_globel_yield_temp,best_globel_output_temp,best_globel_yield_temp2,best_globel_output_temp2

    import numpy as np
    MAE1=np.mean(abs(best_globel_output_temp-best_globel_yield_temp))
    MAE2=np.mean(abs(best_globel_output_temp2-best_globel_yield_temp2))
    MAE=(MAE1+MAE2)/2
    
    MAE1_per=MAE1/np.mean(best_globel_yield_temp)
    MAE2_per=MAE2/np.mean(best_globel_yield_temp2)
    MAE_per=(MAE1_per+MAE2_per)/2
    
    '''
    from sklearn.metrics import r2_score
    r2_1 = r2_score(best_globel_output_temp,best_globel_yield_temp)
    r2_2 = r2_score(best_globel_output_temp2,best_globel_yield_temp2)
    r2=(r2_1+r2_2)/2
    '''
    
    
    z = np.polyfit(best_globel_output_temp, best_globel_yield_temp, 1)
    p = np.poly1d(z)
    yhat = p(best_globel_output_temp)
    ybar = sum(best_globel_yield_temp)/len(best_globel_yield_temp)
    SST = sum((best_globel_yield_temp - ybar)**2)
    SSreg = sum((yhat - ybar)**2)
    R2 = SSreg/SST
    
    print('MAE: ',round(MAE,0),', MAE%: ',round(MAE_per,3), ', R2: ',round(R2,2))
    
    # ------------------------save best prediction ----------------------------------------
    '''
    dict_temp={}
    for key in ['best_globel_yield_loss','best_globel_yield_temp','best_globel_output_temp', 'best_globel_outputAll_temp', 'best_globel_yield_temp2', 'best_globel_output_temp2', 'best_globel_outputAll_temp2']:
        if (str(type(globals()[key]))!= "<class 'module'>") and (str(type(globals()[key]))!= "<class 'function'>")and (str(type(globals()[key]))!= "<class 'dict'>"):
            dict_temp[key]=globals()[key]
            
    with open('dict_'+str(testing_years[0])+'_Skernels'+str(soil_kernel_1)+'-'+str(soil_kernel_2)+
              '_Wkernels'+str(weather_kernel_1)+'-'+str(weather_kernel_2)+'S'+str(int(S_CNN_flag))+'W'+
              str(int(W_CNN_flag))+'TF'+str(int(TF_test_flag))+str(int(TF_train_flag))+
              'S2Y'+str(int(S2Y_flag))+'GRU'+str(int(GRU_flag))+str(int(no_GRU_NDVI_flag))+str(int(no_GRU_NDVI_W_flag))+dt.now().strftime("_%m_%d_%Y_%H_%M_%S")+'.pickle',"wb") as f:
        pickle.dump(dict_temp, f)
    '''        

    return round(MAE,0),round(MAE_per,3),round(R2,2)

#------------------------ program runing --------------------------------------------------------------------------
def program_run_with_BO_training():
        
    if (kernel_number_test_flag==False) and (component_test_flag==False):
        
        print('start! '+dt.now().strftime("%m/%d/%Y,%H:%M:%S"), flush = True)
        print_globals_setting()
        
        #dataset_building()
        bayesopt = BayesianOptimization(train_BO, params)
        bayesopt.maximize(init_points=5, n_iter=BO_n_iter)
                    
        print_accuracy()
        print('Done! '+dt.now().strftime("%m/%d/%Y,%H:%M:%S"), flush = True)
        
    else:
        print('The flags of kernel_number and component are not both False!',flush = True)


def program_run_kernel_component_change(BO_params_path):
    
    global GRU_flag,kernel_number_test_flag,component_test_flag
    
    if (kernel_number_test_flag==False) and (component_test_flag==False):
        print('The flags of kernel_number or component are both False!',flush = True)
        return
    
    print('start!'+dt.now().strftime("%m/%d/%Y,%H:%M:%S"), flush = True)
    print_globals_setting()
    
    f = open(BO_params_path,"rb")
    BO_params=pickle.load(f)
    f.close()
    

    
    if GRU_flag==True:
        #dataset_building()
        _,_,_,_,_,_,_,_ = get_scores(BO_params,test_flag=False,save_flag=False)
    if GRU_flag==False:
        xgboost_model()
        
    MAE,MAE_per,R2=print_accuracy()
    print('Done!'+dt.now().strftime("%m/%d/%Y,%H:%M:%S"), flush = True)
    
    return MAE,MAE_per,R2
       
 

# ----------------------------- train the original model with BO --------------------------------------
# change the training years and testing years
'''
hyperparameters_default()
hyperparameters_setting(['training_years','testing_years'],[['2019','2018'],['2017']])
program_run_with_BO_training()
'''
'''


# ----------------------------- Kernel numbers change --------------------------------------

soil_kernel_changes=['soil_kernel_1','soil_kernel_2']
soil_kernel_values=[[2,8],[8,8],[4,2],[4,4],[4,16]]
weather_kernel_changes=['weather_kernel_1','weather_kernel_2']
weather_kernel_values=[[16,12],[4,12],[8,8],[8,16]]


for year in ['2018','2019W']:
    
    hyperparameters_default()
        
    for kernel_values_change in soil_kernel_values:
        #ipdb.set_trace()
        hyperparameters_default()
        best_globel__default()
        hyperparameters_setting(soil_kernel_changes,kernel_values_change)
        hyperparameters_setting(['setted_epochs_num'],[10])
        
        if year=='2017':
            hyperparameters_setting(['training_years','testing_years'],[['2018','2019'],['2017']])
            BO_path='2017_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2018':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2019'],['2018']])
            BO_path='2018_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2019':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2018'],['2019']])
            BO_path='2019_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2019W':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2018','2019'],['2019W']])
            BO_path='2019W_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
            
        program_run_kernel_component_change(BO_path)
    
    for kernel_values_change in weather_kernel_values:
        #ipdb.set_trace()
        hyperparameters_default()
        best_globel__default()
        hyperparameters_setting(['setted_epochs_num'],[10])
        hyperparameters_setting(weather_kernel_changes,kernel_values_change)
        
        if year=='2017':
            hyperparameters_setting(['training_years','testing_years'],[['2018','2019'],['2017']])
            BO_path='2017_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2018':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2019'],['2018']])
            BO_path='2018_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2019':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2018'],['2019']])
            BO_path='2019_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2019w':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2018','2019'],['2019W']])
            BO_path='2019W_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'

        program_run_kernel_component_change(BO_path)


# ----------------------------- Components change --------------------------------------

components_set=['S_CNN_flag','W_CNN_flag','TF_test_flag','TF_train_flag','S2Y_flag','GRU_flag','no_GRU_NDVI_flag','no_GRU_NDVI_W_flag']

for year in ['2018','2019W']:
    
    hyperparameters_default()
    
    for component_selected in components_set:
        
        hyperparameters_default()
        best_globel__default()
        hyperparameters_setting(['setted_epochs_num'],[10])
        
        if year=='2017':
            hyperparameters_setting(['training_years','testing_years'],[['2018','2019'],['2017']])
            BO_path='2017_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2018':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2019'],['2018']])
            BO_path='2018_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2019':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2018'],['2019']])
            BO_path='2019_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'
        if year=='2019W':
            hyperparameters_setting(['training_years','testing_years'],[['2017','2018','2019'],['2019W']])
            BO_path='2019W_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111.pkl'

        
        if (year=='2019W') and (component_selected=='no_GRU_NDVI_flag'):
            continue
        
        if (component_selected=='no_GRU_NDVI_W_flag'):
            component_selected=[component_selected,'GRU_flag','no_GRU_NDVI_flag']
            hyperparameters_setting(component_selected,[False,False,False])
        elif (component_selected=='no_GRU_NDVI_flag') :
            component_selected=[component_selected,'GRU_flag']
            hyperparameters_setting(component_selected,[False,False])
        else:
            hyperparameters_setting([component_selected],[False])
            
        program_run_kernel_component_change(BO_path)


'''

#------------------------------------- For test only --------------------------------------------------------

'''
hyperparameters_default()


path='D:/quantification_growth_yield_prediction/5142021results/'
#path='./'
#name='2019Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111'
name='2019_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111'
#name='2018Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111'
#name='2017_Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111'


f = open(path+name+'.pkl',"rb")
BO_params=pickle.load(f)
f.close()


import pickle
dict_temp=pickle.load(open( path+'dict_'+name+'.pickle', "rb" ))
best_globel_yield_loss = dict_temp['best_globel_yield_loss']
best_globel_output_temp=dict_temp['best_globel_output_temp']
best_globel_yield_temp=dict_temp['best_globel_yield_temp']
best_globel_outputAll_temp=dict_temp['best_globel_outputAll_temp']
best_globel_output_temp2=dict_temp['best_globel_output_temp2']
best_globel_yield_temp2=dict_temp['best_globel_yield_temp2']
best_globel_outputAll_temp2=dict_temp['best_globel_outputAll_temp2']

# concat, scale back and attached the prediction with the original GPS
hyperparameters_setting(['training_years','testing_years'],[['2019','2018'],['2017']])
hyperparameters_setting(['pre_model_parameters_path'],[name+'.pt'])
batch_size=BO_params['batch_size']
dataset_building(batch_size)
cut_index=len(best_globel_output_temp2)-(len(globals()['img_yield_'+testing_years[0]])%batch_size)
whole_output=np.concatenate((best_globel_output_temp,best_globel_output_temp2[cut_index:]),axis=0)
whole_yield=np.concatenate((best_globel_yield_temp,best_globel_yield_temp2[cut_index:]),axis=0)
best_globel_outputAll_temp=best_globel_outputAll_temp[1:,0:4]
best_globel_outputAll_temp2=best_globel_outputAll_temp2[0:-1,0:4]
whole_outputAll=np.concatenate((best_globel_outputAll_temp,best_globel_outputAll_temp2[cut_index:]),axis=0)


#whole_outputAll[:,0]=globals()['scaler_07_'+testing_years[0]].inverse_transform(whole_outputAll[:,0])
#whole_outputAll[:,1]=globals()['scaler_08_'+testing_years[0]].inverse_transform(whole_outputAll[:,1])
#whole_outputAll[:,2]=globals()['scaler_08_'+testing_years[0]].inverse_transform(whole_outputAll[:,2])

t3=pd.concat([globals()['img_yield_'+testing_years[0]],pd.DataFrame(whole_output,index=globals()['img_yield_'+testing_years[0]].index,columns=['predict']),
              pd.DataFrame(whole_yield,index=globals()['img_yield_'+testing_years[0]].index,columns=['yield']),
              pd.DataFrame(whole_outputAll,index=globals()['img_yield_'+testing_years[0]].index)],axis=1)
output_with_row_num=pd.DataFrame([])
for i in range(len(row_col_E)):
    output_with_row_num=output_with_row_num.append(pd.concat([t3[t3.index==i],pd.DataFrame(row_col_E.iloc[i,:]).transpose()],axis=1))

# figure drawing
img_08_2019_plot=np.resize(img_08_2019,(63,37))
output_with_row_num_07=np.resize(output_with_row_num.iloc[:,-8],(63,38))
#output_with_row_num_07=np.nan_to_num(output_with_row_num_07,nan=np.nanmin(output_with_row_num_07))
plt.imshow(output_with_row_num_07,cmap='jet',vmin=200, vmax=4500)
#plt.imshow(img_08_2019_plot,cmap='jet',vmin=-1, vmax=1)
plt.axis('off')
plt.colorbar()
plt.savefig('filename.png', dpi=300)
'''
'''
# used the trained model for new prediction
hyperparameters_setting(['training_years','testing_years'],[['2017','2018'],['2019']])
hyperparameters_setting(['pre_model_parameters_path'],[path+name+'.pt'])
best_globel__default()
R2,yield_loss,_,_,_,_,_,_=get_scores(BO_params,test_flag=True)
print_globals_setting()
print_accuracy()


import matplotlib.pyplot as plt
plt.scatter(best_globel_output_temp2, best_globel_yield_temp2, c="b", alpha=0.5)
plt.yticks([0,1000,2000,3000,4000,5000])
plt.xticks([0,1000,2000,3000,4000,5000])
'''

#------------------ predict for future, partly weather and image data avalible--------------------------------------------------------
replacement_set=[25,24,23,22,21,20,19,18,17,16,15,14]

hyperparameters_default()


path='D:/quantification_growth_yield_prediction/5142021results/'
#path='./'

for year in ['2017','2018','2019']:

    name=year+'Skernels4-8_Wkernels8-12S1W1TF11S2Y1GRU111'
    
    
    f = open(path+name+'.pkl',"rb")
    BO_params=pickle.load(f)
    f.close()
    
    hyperparameters_setting(['testing_years'],[[year]])
    hyperparameters_setting(['pre_model_parameters_path'],[path+name+'.pt'])
    
    
    temp1=globals()['weather'+testing_years[0]].detach().clone()
    temp2=globals()['irrigation_data_'+testing_years[0]].copy()
    temp3=globals()['img_yield_'+testing_years[0]].copy()
    
    week_number_irrigation=[]  
    if globals()['irrigation_data_'+testing_years[0]].size>40: # if no irrigation data, dimension is 20*2
        for irrigation_date in globals()['irrigation_data_'+testing_years[0]].columns[2:]:
            week_number_irrigation.append(math.ceil((int(irrigation_date)+plantting_date[testing_years[0]])/7))
            
   
    for replace in replacement_set:
        #print(replace)
        
        # weather data replacement
        globals()['weather'+testing_years[0]][:,:,replace:]=globals()['weather8years'][:,:,replace:]
        
        # remove the future irrigation data
        for i in range(len(week_number_irrigation)):
            #print(i)
            if replace<=week_number_irrigation[i]:
                globals()['irrigation_data_'+testing_years[0]]=globals()['irrigation_data_'+testing_years[0]].iloc[:,:i+2]
        
        # remove the future image data
        a = np.empty((len(globals()['img_yield_'+testing_years[0]]),1))
        a[:] = -100
        if year=='2017':
            if replace<=16: # image of 8/12/2017 is not avalible
                globals()['img_yield_'+testing_years[0]].iloc[:,14:16]=np.repeat(a,2,axis=1)
        if year=='2018':
            if replace<=18: # image of 8/22/2018 and 9/15/2018 are not avalible
                globals()['img_yield_'+testing_years[0]].iloc[:,14:16]=np.repeat(a,2,axis=1)
            elif replace<=21: # image of 9/15/2018 is not avalible
                globals()['img_yield_'+testing_years[0]].iloc[:,15:16]=a
        if year=='2019':
            if replace<=16: # image of 8/14/2019 and 9/6/2019 are not avalible
                globals()['img_yield_'+testing_years[0]].iloc[:,14:16]=np.repeat(a,2,axis=1)
            elif replace<=20: # image of 9/6/2019 is not avalible
                globals()['img_yield_'+testing_years[0]].iloc[:,15:16]=a


        print('replace the weather, irrigation and image data from week '+str(replace))    
        best_globel__default()
        R2,yield_loss,_,_,_,_,_,_=get_scores(BO_params,test_flag=True)
        print_globals_setting()
        print_accuracy()
        
        # give back the original data
        globals()['weather'+testing_years[0]]=temp1.detach().clone()
        globals()['irrigation_data_'+testing_years[0]]=temp2.copy()
        globals()['img_yield_'+testing_years[0]]=temp3.copy()
    
    