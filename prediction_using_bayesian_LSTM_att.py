# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:49:55 2022
@author: NickWang
"""
## import packages
import numpy as np
import pandas as pd
#Pandas是基于Numpy的专业数据分析工具, 可以灵活高效的处理各种数据集
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import tqdm
#python进度条库，tqdm加载一个可迭代对象，并以进度条的形式实时显示该可迭代对象的加载进度。
import random
import os

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *#获取库中所有的函数
from tensorflow.keras import backend as K
#from keras.layers import Permute
from math import sqrt

import scipy.io
omni = pd.read_hdf(r'H:\Search\DST_prediction\omni_hourly_index_1998_2020.h5')

#cols = ['time','btot','bx','by','bz','speedx','tem','den','pdyn','spot','dst']
cols = ['time','btot','bx','bygsm','bzgsm','speed','speedx','tem','den','pdyn','Electric','dst']
omni = omni[cols]
timei = omni['time'].values
#时间单独用于画图
dst = omni['dst']

laghead = 12
omni_in = omni[0:-laghead]
dst_lag = dst[laghead:]
omni_in = omni_in.drop(['time'],axis=1)
t_in = timei[0:-laghead]
#plt.figure(figsize=(20,10))
#omni['dst'].plot()
tstart = '20140101'
tmid = '20160101'
tend = '20180101'

numstart = mdate.date2num(datetime.strptime(tstart,'%Y%m%d'))
numend = mdate.date2num(datetime.strptime(tend,'%Y%m%d'))
nummid = mdate.date2num(datetime.strptime(tmid,'%Y%m%d'))

val_id = (t_in>=numstart)&(t_in<nummid)#14,15年的数据,校验数据
test_id = (t_in>=nummid)&(t_in<=numend)#16,17年的数据作为测试数据
train_id = (t_in<numstart)|(t_in>numend)#其他年份的数据全部用于训练
train = omni_in[train_id]
val = omni_in[val_id]
test = omni_in[test_id]
#val_time = t_in[val_id]
#test_time = t_in[test_id]
dst_train = dst_lag[train_id]
dst_val = dst_lag[val_id]
dst_test = dst_lag[test_id]
#dstslice = dst[(timei>=numstart)&(timei<numend)]

def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
"""
def to_sequences_MD (seq_size,obs,xcols,ycols, step=1):
    x = []
    y = []
    y_time_id = []
    
    for i in range(0,len(obs)-SEQUENCE_SIZE,step):
    
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)][xcols]
        after_window = obs[(i+SEQUENCE_SIZE-1):(i+SEQUENCE_SIZE)][ycols]
        win_id = after_window.index[0]
        #window = [[x] for x in window]
        
        win_values = window.values
        #print("{} - {}".format(window,after_window))
        x.append(win_values)
        y.append(after_window.values)
        y_time_id.append(win_id)
        
    return np.array(x),np.array(y),y_time_id
"""
def to_sequences_MD_lag (seq_size,obs,yout,xcols, step=1):
    x = []
    y = []
    y_time_id = []
    
    for i in range(0,len(obs)-SEQUENCE_SIZE,step):
        #SEQUENCE_SIZE为滑窗长度，防止训练集选取时出现不完整
        '''
        seq_size: the number of observations will be used for training, time step
        step: the step of time moving window, change of observations to form a new training data, default is 1 
        obs: the original dataset
        xcols: the column names of X
        ycols: the column names of Y
        '''
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)][xcols]
        after_window = yout[(i+SEQUENCE_SIZE):(i+SEQUENCE_SIZE+1)]
        win_id = after_window.index[0]
        #window = [[x] for x in window]
        
        win_values = window.values
        #print("{} - {}".format(window,after_window))
        x.append(win_values)
        y.append(after_window.values)
        y_time_id.append(win_id)
        
    return np.array(x),np.array(y),y_time_id
#our target is dst (Y), the bsalar, 'bx','bygsm','bzgsm','speed','speedx',
#and 'den' will used for external variables (X)

## I have about 20 years data and 0.8 for training, thus set date 2016
'''
train_size = int(len(omni_in) * 0.9)
test_size = len(omni_in) - train_size
train, test = omni_in[0:train_size], omni_in[train_size:]
dst_train,dst_test = dst_lag[:train_size],dst_lag[train_size:]
'''
SEQUENCE_SIZE = 24 ##timestep

xcol = ['dst']
ycol = ['dst']
#x_train,y_train,train_time_id = to_sequences_MD(SEQUENCE_SIZE,train, xcols = xcol,ycols = ycol)
x_train,y_train,train_time_id = to_sequences_MD_lag(SEQUENCE_SIZE,train, dst_train,xcols = xcol)
x_val,y_val,val_time_id = to_sequences_MD_lag(SEQUENCE_SIZE,val, dst_val,xcols = xcol)
x_test,y_test,test_time_id = to_sequences_MD_lag(SEQUENCE_SIZE,test,dst_test,xcols =xcol)


df_x_train = np.asarray(x_train).reshape(-1,SEQUENCE_SIZE,len(xcol))
df_x_val = np.asarray(x_val).reshape(-1,SEQUENCE_SIZE,len(xcol))
df_x_test = np.asarray(x_test).reshape(-1,SEQUENCE_SIZE,len(xcol))
#此处将数据转化为三维的

#train_time = timei[train_time_id]
#test_time = timei[test_time_id]
train_time = timei[train_time_id]
val_time = timei[val_time_id]
test_time = timei[test_time_id]

df_y_train = np.asarray(y_train).reshape(-1,1)
df_y_val = np.asarray(y_val).reshape(-1,1)
df_y_test = np.asarray(y_test).reshape(-1,1)

print("Shape of x training set: {}".format(df_x_train.shape))
print("Shape of y test set: {}".format(df_y_test.shape))

print("Shape of x holdout set: {}".format(df_x_val.shape))
print("Shape of y holdout set: {}".format(df_y_val.shape))

### CONCATENATE TRAIN/TEST DATA AND LABEL ### 
X = np.concatenate([df_x_train,df_x_val,df_x_test],axis=0)
y = np.concatenate([df_y_train,df_y_val,df_y_test],axis=0)

print(X.shape,y.shape)


scipy.io.savemat('mod1_data12.mat',{'train_time':train_time,'val_time':val_time,'test_time':test_time,\
                                    'df_x_train':df_x_train,'df_x_val':df_x_val,'df_x_test':df_x_test,\
                 'df_y_train':df_y_train,'df_y_val':df_y_val,'df_y_test':df_y_test})
    #将数据存储在.mat文件中，以便在不同应用程序或操作系统之间共享数据
xcols =  ['btot','bx','bygsm','bzgsm','speed','speedx','tem','den','pdyn','Electric']
ycols = ['dst']

#ext_x_train,_ ,_ = to_sequences_MD(SEQUENCE_SIZE,train,xcols = xcols, ycols = ycols)
#ext_x_test,_ ,_ = to_sequences_MD(SEQUENCE_SIZE,test,xcols = xcols, ycols = ycols)

ext_x_train,ext_y_train,_ = to_sequences_MD_lag(SEQUENCE_SIZE,train,dst_train,xcols = xcols)
ext_x_val,ext_y_val,_ = to_sequences_MD_lag(SEQUENCE_SIZE,val,dst_val,xcols = xcols)
ext_x_test,ext_y_test,_ = to_sequences_MD_lag(SEQUENCE_SIZE,test,dst_test,xcols = xcols)
## convert to np array
ext_x_train = np.asarray(ext_x_train).reshape(-1,SEQUENCE_SIZE,len(xcols))
ext_x_val = np.asarray(ext_x_val).reshape(-1,SEQUENCE_SIZE,len(xcols))
ext_x_test = np.asarray(ext_x_test).reshape(-1,SEQUENCE_SIZE,len(xcols))


print("Shape of x training set: {}".format(ext_x_train.shape))
print("Shape of y training set: {}".format(ext_y_train.shape))

print("Shape of x holdout set: {}".format(ext_x_val.shape))
print("Shape of y holdout set: {}".format(ext_y_val.shape))



### CONCATENATE TRAIN/TEST EXTERNAL FEATURES ###
Ext_features = np.concatenate([ext_x_train,ext_x_val,ext_x_test],axis=0)

print(Ext_features.shape)

scipy.io.savemat('mod1_ext_data06.mat',{'ext_x_train':ext_x_train,'ext_x_val':ext_x_val,'ext_x_test':ext_x_test,\
                 "ext_y_train":ext_y_train,'ext_y_val':ext_y_val,'ext_y_test':ext_y_test})
    
## 5.1 Model 1 - autoencoder and LSTM prediction

## 1 Autoencoder and fit - explain the misspecification part

#The model first primes the network by auto feature extraction,
#training an LSTM Autoencoder, which is critical to capture complex time-series dynamics at scale.
model_name = 'Dstout_MC_lstm_mod_t61'
#model_save_path1 = './mymodel/' + model_name +'/autocoder'+ '/auto_model.h5'
#model_save_path2 = './mymodel/' + model_name +'/lstm'+ '/lstm_model.h5'
model_weights_path1 = './mymodel/' + model_name +'/autocoder'+ '/cp.ckpt'
model_weights_path2 = './mymodel/' + model_name +'/lstm'+ '/cp.ckpt'

early_stopping_cb =tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
'''

attention_mul = attention_3d_block2(encoded_ae)
attention_mul = Flatten()(attention_mul,training = True)
L = RepeatVector(SEQUENCE_SIZE)(attention_mul,training=True)
decoded_ae1 = attention_3d_block2(L)
decoded_ae1 = Bidirectional(LSTM(20, return_sequences=True, dropout=0.3))(decoded_ae1, training=True)
decoded_ae = Bidirectional(LSTM(20, return_sequences=True, dropout=0.3))(decoded_ae1, training=True)
out_ae = TimeDistributed(Dense(1))(decoded_ae)
'''
inputs_ae = Input(shape=(SEQUENCE_SIZE, 1))
encoded_ae1 = Bidirectional(LSTM(20, return_sequences=True, dropout=0.3))(inputs_ae, training=True)
encoded_ae = Bidirectional(LSTM(15, return_sequences=True, dropout=0.3))(encoded_ae1, training=True)

decoded_ae1 = Bidirectional(LSTM(20, return_sequences=True, dropout=0.3))(encoded_ae, training=True)
decoded_ae =Bidirectional(LSTM(15, return_sequences=True, dropout=0.3))(decoded_ae1, training=True)
out_ae = TimeDistributed(Dense(1))(decoded_ae)


sequence_autoencoder = Model(inputs_ae, out_ae)
sequence_autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
#损失函数和评估方法都是mse

### TRAIN AUTOENCODER ###
sequence_autoencoder.fit(X[:len(df_x_train)], X[:len(df_x_train)], validation_data=(df_x_val,df_x_val),batch_size=1024, epochs=100, verbose=2, shuffle = True,callbacks=early_stopping_cb)
sequence_autoencoder.save_weights(model_weights_path1)
#sequence_autoencoder.save(model_save_path1)
#2 Encode EVI - LSTM and concatenate external features (regressors)
#Features vectors are then concatenated with the new input and fed to LSTM Forecaster for
#prediction.

## encode the original X and later this information will be used for as input
#X = np.concatenate([df_x_train,df_x_val,df_x_test],axis=0) 
encoder = Model(inputs_ae, encoded_ae)
XX = encoder.predict(X)
#XX就是X的预测值，其中X是监督学习的基本数据
XXE = np.concatenate([XX, Ext_features], axis=2)
#Ext_features = np.concatenate([ext_x_train,ext_x_val,ext_x_test],axis=0)
#ext是外部变量，df为主要的预测内容，即DST。
XXE.shape

#The above results will be our input for the prediction model

#SPLIT TRAIN TEST 
X_train = XXE[:len(df_x_train)]
X_test = XXE[-len(df_x_test):]
y_train = y[:len(df_y_train)]
y_test = y[-len(df_y_train):]
#print(X_train.shape)

##hold out data
#val_size = int(len(X_train)*0.1)

X_val = XXE[len(df_x_train):len(df_x_train)+len(df_x_val)]
y_val = y[len(df_y_train):len(df_y_train)+len(df_y_val)]
#print(X_val,y_val)
#print(X_val.shape, y_val.shape)
### Scaling the data based on the variables or features (each features needs to scale)
#first need to convert into 2D (n_samples, n_features) and then change it back

### SCALE DATA ##
scaler1 = StandardScaler()
#将所有数据的特征值转换为均值为0，而方差为1的状态
X_train = scaler1.fit_transform(X_train.reshape(-1,XXE.shape[-1])).reshape(-1,SEQUENCE_SIZE,XXE.shape[-1])
X_test = scaler1.transform(X_test.reshape(-1,XXE.shape[-1])).reshape(-1,SEQUENCE_SIZE,XXE.shape[-1])
X_val = scaler1.transform(X_val.reshape(-1,XXE.shape[-1])).reshape(-1,SEQUENCE_SIZE,XXE.shape[-1])
tf.random.set_seed(1)
### DEFINE FORECASTER ###
inputs1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
conv1 = Conv1D(filters=8, kernel_size=1)(inputs1,training=True)
Dropout(0.3)(conv1)
lstm1 = Bidirectional(LSTM(20, return_sequences=True, dropout=0.3))(conv1, training=True)
attention_mul = attention_3d_block2(lstm1)
attention_mul = Flatten()(attention_mul)
#lstm1 = LSTM(15, return_sequences=False, dropout=0.3)(lstm1, training=True)
dense1 = Dense(10)(attention_mul)
out1 = Dense(1)(dense1)

model1 = Model(inputs1, out1)
model1.compile(loss='mse', optimizer='adam', metrics=['mse'])

### FIT FORECASTER ### validation_split=0.1 #validation_data=(X_val, y_val)
history = model1.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=100, batch_size=1024, verbose=2, shuffle = True, callbacks=early_stopping_cb)
model1.save_weights(model_weights_path2)
#model1.save(model_save_path2)
p = 0.05
dropout_rate = 0.3
### FUNCTION FOR STOCHASTIC DROPOUT FOR SINGLE LOCATION ###

def test_stoc_drop1(df_x_test,ext_x_test, R, enc, NN): #for model 1
    
    X,  F = df_x_test,ext_x_test
    
    enc_pred = np.vstack(enc([X, R]))
    enc_pred = np.concatenate([enc_pred, F], axis=2)
    trans_pred = scaler1.transform(enc_pred.reshape(-1,enc_pred.shape[-1])).reshape(-1,SEQUENCE_SIZE,enc_pred.shape[-1])
    NN_pred = NN([trans_pred, R])
    
    return np.vstack(NN_pred)

'''
mae1_test = []
enc = K.function([encoder.layers[0].input], [encoder.layers[-1].output])
NN = K.function([model1.layers[0].input], [model1.layers[-1].output])
for i in tqdm.tqdm(range(0,100)):
    mae1_test.append(mean_absolute_error(df_y_test, test_stoc_drop1(df_x_test,ext_x_test, R = dropout_rate, enc = enc, NN = NN)))
    
print(np.mean(mae1_test), np.std(mae1_test))
'''
### COMPUTE STOCHASTIC DROPOUT FOR SINGLE COUNTY ###

def holdout_uncertainty_model11(df_x_test,ext_x_test,df_y_test,p,dropout_rate):
    
    pred1 = []
   
   #  r_test = []
    enc = K.function([encoder.layers[0].input], [encoder.layers[-1].output])
    NN = K.function([model1.layers[0].input], [model1.layers[-1].output])

    for i in tqdm.tqdm(range(0,100)):
        
        pred1_test = test_stoc_drop1(df_x_test,ext_x_test, R = dropout_rate, enc = enc, NN = NN)
                             
        pred1.append(pred1_test)
       
#        mae_test.append(mean_absolute_error(df_y_test, pred1_test))
#        mse_test.append(mean_squared_error(df_y_test, pred1_test))
    pred1 = np.asarray(pred1)
    pp = pred1[:,:,0].T ##delete one dimension just assign one dimesion to 0
    
    lower = np.quantile(pp,p/2,axis =1)
    upper = np.quantile(pp,1- p/2,axis =1)
    medium = np.quantile(pp,0.5,axis =1)
   

    fdf = pd.DataFrame({
        "true":df_y_test.ravel(),
        "lower":lower,
        "medium":medium,
        "upper":upper
        })
    
    return fdf,pp


'''

XE = np.concatenate([X, Ext_features], axis=2)
print(XE.shape)

### SPLIT TRAIN TEST ###
X_train2, X_test2 = XE[:len(df_x_train)], XE[len(df_x_train):]
y_train2, y_test2 = y[:len(df_x_train)], y[len(df_x_train):]

### SCALE DATA ###
scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2.reshape(-1,X_train2.shape[-1])).reshape(-1,SEQUENCE_SIZE,X_train2.shape[-1])
X_test2 = scaler2.transform(X_test2.reshape(-1,X_train2.shape[-1])).reshape(-1,SEQUENCE_SIZE,X_train2.shape[-1])

tf.random.set_seed(13)
### DEFINE LSTM FORECASTER ###
inputs2 = Input(shape=(X_train2.shape[1], X_train2.shape[2]))
lstm2 = LSTM(128, return_sequences=True, dropout=0.3)(inputs2, training=True)
lstm2 = LSTM(32, return_sequences=False, dropout=0.3)(lstm2, training=True)
dense2 = Dense(50)(lstm2)
out2 = Dense(1)(dense2)

model2 = Model(inputs2, out2)
model2.compile(loss='mse', optimizer='adam', metrics=['mse'])

history = model2.fit(X_train2, y_train2, validation_split=0.1, epochs=100, batch_size=128, verbose=2, shuffle = True, callbacks=early_stopping_cb)
#model2.save_weights(model_weights_path2)
'''
def test_stoc_drop2(df_x_test,ext_x_test, R, NN): #for model 2
    
    X,  F = df_x_test,ext_x_test
    XF = np.concatenate([X, F], axis=2)
    trans_pred = scaler2.transform(XF.reshape(-1,XF.shape[-1])).reshape(-1,SEQUENCE_SIZE,XF.shape[-1])
    NN_pred = NN([trans_pred, R])
    return np.vstack(NN_pred)

def holdout_uncertainty_model12(df_x_test,ext_x_test,df_y_test,p,dropout_rate):
    
    pred1 = []
    mae_test = []
   
   # rmse_test = []
   # r_test = []
#    enc = K.function([encoder.layers[0].input], [encoder.layers[-1].output])
    NN = K.function([model2.layers[0].input], [model2.layers[-1].output])

    for i in tqdm.tqdm(range(0,100)):
        
        pred1_test = test_stoc_drop2(df_x_test,ext_x_test, R = dropout_rate, NN = NN)
                             
        pred1.append(pred1_test)
        
        mae_test.append(mean_absolute_error(df_y_test, pred1_test))
      
    pred1 = np.asarray(pred1)
    pp = pred1[:,:,0].T ##delete one dimension just assign one dimesion to 0
    lower = np.quantile(pp,p/2,axis =1)
    upper = np.quantile(pp,1- p/2,axis =1)
    medium = np.quantile(pp,0.5,axis =1)

    fdf = pd.DataFrame({
        "true":df_y_test.ravel(),
        "lower":lower,
        "medium":medium,
        "upper":upper
        })
    
    return fdf,mae_test

def metrics(pp,y_obs):
    mae = []
    msem = []
    msen = []
    mpp = np.mean(pp,axis=1).reshape(-1,1)
    repmpp = np.tile(mpp,[1,pp.shape[1]])
    y_val = np.tile(y_obs,[1,100])
    for i in range(pp.shape[0]):
        msem.append(mean_squared_error(pp[i,:],repmpp[i,:]))
        msen.append(mean_squared_error(pp[i,:],y_val[i,:]))
        mae.append(mean_absolute_error(pp[i,:],y_val[i,:]))
    return mae,msem,msen

def plot_figure_un(x,yact,ypred,sigma):
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot_date(x,yact,color='#CC4F1B',linestyle='dashdot')
    ax1.plot_date(x,ypred,color = '#3F7F4C',linestyle='dashdot')
    ax1.grid()
    ax1.fill_between(x,ypred-sigma,ypred+sigma,alpha=0.2,edgecolor='#aabbcc',facecolor='#089FFF',
    linewidth=4, linestyle='dashdot', antialiased=True)
    ax1.set_xlim(x[0],x[-1])
    ax1.legend(('True shift','Predicted'))
    ax1.set_ylabel('Dst/nT')
    ax1.set_xlabel('time index')
    
    exce_dw = ypred-sigma-yact
    exce_dw[exce_dw<0]=0
    exce_up = yact-ypred-sigma
    exce_up[exce_up<0]=0
    exce = np.vstack([exce_up,exce_dw]).T
    exce_df = pd.DataFrame(exce)
    exce_df.columns = ['exceed upper','exceed lower']
    
    ax2 = fig.add_subplot(2,1,2)
    exce_df.plot.area(stacked = False,alpha = 0.2,ax=ax2)
    #ax2.set_xlim(-6,145)
    #ax2.set_xticks(np.arange(0,145,24))
    #ax2.set_xticklabels(['12-19','12-20','12-21','12-22','12-23','12-24','12-25'])
    ax2.grid()

    
stin = '20150312' 
stend = '20150322'

stin = '20151219'
stend = '20151225'

#stin = '20170526'
#stend = '20170531'

numstart1 = mdate.date2num(datetime.strptime(stin,'%Y%m%d'))
numend1 = mdate.date2num(datetime.strptime(stend,'%Y%m%d'))
#st_in_size = (in_time>=numstart1)&(in_time<numend1)
st_out_size = (val_time>=numstart1)&(val_time<numend1)
st_time = val_time[st_out_size]
x_val_slice = df_x_val[st_out_size]
ext_x_val_slice = ext_x_val[st_out_size]
y_val_slice = df_y_val[st_out_size]

holdout_uncer11, pp = holdout_uncertainty_model11(x_val_slice,ext_x_val_slice,y_val_slice,p,dropout_rate)
#holdout_uncer11, pp  = holdout_uncertainty_model11(df_x_val,ext_x_val,df_y_val,p,dropout_rate)

pplag = pp[laghead:]
#y_val_lag = df_y_val[:-laghead]
y_val_lag = y_val_slice[:-laghead]
mae,msem,msen = metrics(pplag,y_val_lag)

maearr =  np.asarray(mae)
print(np.mean(maearr), np.std(maearr))

msemarr = np.asarray(msem)
msenarr = np.asarray(msen)
al = np.std(np.sqrt(msemarr))
ep = np.std(np.sqrt(msenarr))
sigma = np.std(np.sqrt(msemarr+msenarr))
#print(np.mean(msemarr),np.std(msemarr))
#print(np.mean(msenarr),np.std(msenarr))

#print('\n')
#print(np.mean(rmse11), np.mean(r11))

'''
holdout_uncer12, mmae12 = holdout_uncertainty_model12(x_val_slice,ext_x_val_slice,y_val_slice,p,dropout_rate)
print(np.mean(mmae12), np.std(mmae12))

'''
#x = list(holdout_uncer11.index)
x = st_time[:-laghead]
#x = val_time[:-laghead]
act = holdout_uncer11['true'].values
yact = act[:-laghead]
pred = holdout_uncer11['medium'].values
ypred = pred[laghead:]
#pd_act = pd.DataFrame(act)
#act_shift = pd_act.shift(laghead)
plot_figure_un(x,yact,ypred,sigma)

from helper_functions import *

plotcorr(yact, ypred)

stin = '20160305'
stend = '20160310'
stin = '20161012'
stend = '20161017'
stin='20160506'
stend='20160513'
stin = '20170526'
stend = '20170531'
stin = '20170715'
stend = '20170721'
#stin = '20160101'
#stend = '20180101'
numstart1 = mdate.date2num(datetime.strptime(stin,'%Y%m%d'))
numend1 = mdate.date2num(datetime.strptime(stend,'%Y%m%d'))
#st_in_size = (in_time>=numstart1)&(in_time<numend1)
st_out_size = (test_time>=numstart1)&(test_time<numend1)
st_time = test_time[st_out_size]
x_test_slice = df_x_test[st_out_size]
ext_x_test_slice = ext_x_test[st_out_size]
y_test_slice = df_y_test[st_out_size]

holdout_uncer11, pp = holdout_uncertainty_model11(x_test_slice,ext_x_test_slice,y_test_slice,p,dropout_rate)
#holdout_uncer11, pp = holdout_uncertainty_model11(df_x_test,ext_x_test,df_y_test,p,dropout_rate)
pplag = pp[laghead:]
y_test_lag = y_test_slice[:-laghead]
#y_test_lag = df_y_test[:-laghead]
mae,msem,msen = metrics(pplag,y_test_lag)

maearr =  np.asarray(mae)
msemarr = np.asarray(msem)
msenarr = np.asarray(msen)

al = np.std(np.sqrt(msemarr))
ep = np.std(np.sqrt(msenarr))
sigma = np.std(np.sqrt(msemarr+msenarr))
print(np.mean(maearr), np.std(maearr))
print(np.mean(msemarr),np.std(msemarr))
print(np.mean(msenarr),np.std(msenarr))
'''
holdout_uncer12, mmae12 = holdout_uncertainty_model12(x_test_slice,ext_x_test_slice,y_test_slice,p,dropout_rate)
print(np.mean(mmae12), np.std(mmae12))
'''
x = st_time[:-laghead]
#x = test_time[:-laghead]
act = holdout_uncer11['true'].values
yact = act[:-laghead]
pred = holdout_uncer11['medium'].values
ypred = pred[laghead:]

plot_figure_un(x,yact,ypred,sigma)


#plt.legend(('True','True shift','Predicted'))

#plt.title("LSTM + Autoender at storm1 ")

plotcorr(yact, ypred)

### PLOT AVG AND UNCERTAINTY OF RESULTS ###
#bar = plt.bar([0,1], [np.mean(mmae11), np.mean(mmae12)], 
#              yerr=[2.95*np.std(mmae11), 2.95*np.std(mmae12)])
#plt.xticks([0,1], ['model1','model2'], rotation=90)
#bar[0].set_color('cyan'), bar[1].set_color('magenta')

#https://wdc.kugi.kyoto-u.ac.jp/dstdir/index.html


