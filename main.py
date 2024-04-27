# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:49:55 2022

@author: huang
"""

## import packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import tqdm
import random
import os

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score
from sklearn.model_selection import KFold,RepeatedKFold

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
#from keras.layers import Permute
from math import sqrt
import scipy.io
from analy_un import calculate_density
#from mastml.error_analysis import ErrorUtils

def to_sequences_MD_lag (seq_size,obs,yout,xcols, step=1):
    x = []
    y = []
    y_time_id = []
    
    for i in range(0,len(obs)-seq_size,step):
        '''
        seq_size: the number of observations will be used for training, time step
        step: the step of time moving window, change of observations to form a new training data, default is 1 
        obs: the original dataset
        xcols: the column names of X
        ycol: the column nam of Y
        '''
        #print(i)
        window = obs[i:(i+seq_size)][xcols]
        after_window = yout[(i+seq_size):(i+seq_size+1)]
        win_id = after_window.index[0]
        #window = [[x] for x in window]
        
        win_values = window.values
        #print("{} - {}".format(window,after_window))
        x.append(win_values)
        y.append(after_window.values)
        y_time_id.append(win_id)
        
    return np.array(x),np.array(y),y_time_id


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

SEQUENCE_SIZE = 2 ##timestep
xcol = ['dst']
ycol = ['dst']

xcols =  ['btot','bx','bygsm','bzgsm','speed','speedx','tem','den','pdyn','Electric']
ycols = ['dst']

omni = pd.read_hdf(r'E:\tfd\BayLSTM\data\omni_hourly_index_1998_2020.h5')

#cols = ['time','btot','bx','by','bz','speedx','tem','den','pdyn','spot','dst']
cols = ['time','btot','bx','bygsm','bzgsm','speed','speedx','tem','den','pdyn','Electric','dst']
omni = omni[cols]
timei = omni['time'].values
dst = omni['dst']

train_storm_list = [('19980214','19980224'),('19980308','19980319'),('19980317','19980327'),('19980422','19980508'),('19980620','19980630'),\
                    ('19980802','19980812'),('19980818','19980902'),('19981014','19981027'),('19981103','19981119'),\
                        ('19990214','19990224'),('19990412','19990423'),('19990908','19990918'),('19990918','19991031'),\
                        ('19991211','19991216'),('20000207','20000218'),('20000806','20000817'),('20010323','20010404'),\
                        ('20000910','20000922'),('20000927','20001009'),('20001024','20001115'),('20010317','20010327'),\
                        ('20010920','20011007'),('20011017','20011113'),('20011120','20011130'),('20020321','20020328'),\
                        ('20020416','20020426'),('20020728','20020807'),('20020927','20021022'),('20030518','20030606'),\
                        ('20030612','20030622'),('20030815','20030826'),('20031023','20031102'),('20031116','20031126'),('20010323','20010404'), \
                        ('20040330','20040409'),('20040720','20040806'),('20041103','20041114'),('20050106','20050116'),\
                        ('20050609','20050629'),('20060402','20060420'),('20061211','20061221'),\
                        ('20100401','20100415'),('20110802','20110812'),('20110923','20111002'),('20111021','20111031'),\
                        ('20120419','20120430'),('20120707','20120715'),('20130314','20130324'),('20140823','20140902'),\
                        ('20120303','20120313'),('20150618','20150628'),('20151216','20151226'),\
                       ('20161009', '20161019'),('20170523','20170603')]  
    
val_storm_list = [('19980818', '19980902'),('19990412','19990423'),('19990918','19991031'),('20000328', '20000426'),\
                  ('20000711', '20000731'),('20011017','2001113'),('20020830','20020914'),('20040826','20040909'),\
                  ('20050503','20050603'),('20090718','20090728'),('20150930', '20151017'),('20170901','20170911')]

test_storm_list = [('19981102','19981112'),('19990109','19990119'), \
                   ('20000118','20000129'),('20000519','20000530'),('20010323','20010404'), \
                   ('20030525','20030604'),('20030707','20030717'),('20040118','20040128'),('20041103','20041114'), \
                    ('20120926','20121007'),('20130527','20130607'),('20130624','20130704'),('20150313','20150323'), \
                        ('20180822','20180903')]    

#train_storm_list = list(set(tuple(storm_list)) - set(tuple(test_storm_list))) #('20161009', '20161019')

laghead = 3
omni_in = omni[0:-laghead]
dst_lag = dst[laghead:]
#dst_lag.reset_index(drop= True, inplace=True)
omni_in = omni_in.drop(['time'],axis=1)
t_in = timei[0:-laghead]
#plt.figure(figsize=(20,10))
#omni['dst'].plot()

'''
test_storm_list = [['19980622','199800701'],['19981102','19981112'],['19990109','19990118'],['19990413','19990419'], \
                   ['20000116','20000126'],['20000402','20000412'],['20000519','20000528'],['20010326','20010404'], \
                   ['20030526','20030606'],['20030708','20030718'],['20040118','20040127'],['20041104','20041114'], \
                    ['20120910','20121005'],['20130528','20130604'],['20130626','20130714'],['20150311','20150321'], \
                        ['20180822','20180903']] '''

x_train_list = []
y_train_list = []
train_time_list = []

x_ext_train_list = []
y_ext_train_list = []
    
for ii in range(len(train_storm_list)):
    print("第{}个训练磁暴".format(ii))
    ymd2num0 = mdate.date2num(datetime.strptime(train_storm_list[ii][0],'%Y%m%d'))
    ymd2num1 = mdate.date2num(datetime.strptime(train_storm_list[ii][1],'%Y%m%d'))
    idx = (t_in >= ymd2num0)&(t_in <=ymd2num1)
    omni_train = omni_in[idx]
    dst_train = dst_lag[idx]
    x_omni_train,y_dst_train,train_idx = to_sequences_MD_lag(SEQUENCE_SIZE,omni_train, dst_train,xcols = xcol)
    x_train_list.append(x_omni_train)
    y_train_list.append(y_dst_train)
    
    train_time_list.append(train_idx)
    
    x_ext_train,y_ext_train,_ = to_sequences_MD_lag(SEQUENCE_SIZE,omni_train, dst_train,xcols = xcols)
    x_ext_train_list.append(x_ext_train)
    y_ext_train_list.append(y_ext_train)

x_train = x_train_list[0]
y_train = y_train_list[0]
train_time_id = train_time_list[0]
ext_x_train = x_ext_train_list[0]
ext_y_train = y_ext_train_list[0]

for jj in range(1,len(x_train_list)):
    x_train =  np.concatenate((x_train,x_train_list[jj],),axis=0)
    y_train = np.concatenate((y_train,y_train_list[jj],),axis=0)
    train_time_id = np.concatenate((train_time_id,train_time_list[jj]),axis=0)
    
    ext_x_train =  np.concatenate((ext_x_train,x_ext_train_list[jj]),axis=0) 
    ext_y_train = np.concatenate((ext_y_train,y_ext_train_list[jj]),axis=0)

df_x_train = np.asarray(x_train).reshape(-1,SEQUENCE_SIZE,len(xcol))
x_ext_train = np.asarray(ext_x_train).reshape(-1,SEQUENCE_SIZE,len(xcols))
df_y_train = np.asarray(y_train).reshape(-1,1)  

print("Shape of x training set: {}".format(df_x_train.shape))
print("Shape of y test set: {}".format(df_y_train.shape))
 
#scipy.io.savemat('train_storm_arr01.mat',{'x_train_list':x_train_list,'x_ext_train_list':x_ext_train_list,'y_train_list':y_train_list})

x_val_list = []
y_val_list = []
val_time_list = [] 

x_ext_val_list = []
y_ext_val_list = []

for ii in range(len(val_storm_list)):
    print("第{}个验证磁暴".format(ii))
    ymd2num0 = mdate.date2num(datetime.strptime(val_storm_list[ii][0],'%Y%m%d'))
    ymd2num1 = mdate.date2num(datetime.strptime(val_storm_list[ii][1],'%Y%m%d'))
    idx = (t_in >= ymd2num0)&(t_in <ymd2num1)
    omni_val = omni_in[idx]
    dst_val = dst_lag[idx]
   
    x_omni_val,y_dst_val,val_idx = to_sequences_MD_lag(SEQUENCE_SIZE,omni_val, dst_val,xcols = xcol)
    x_val_list.append(x_omni_val)
    y_val_list.append(y_dst_val)
    val_time_list.append(val_idx)
    
    x_ext_val,y_ext_val,_ = to_sequences_MD_lag(SEQUENCE_SIZE,omni_val, dst_val,xcols = xcols)
    x_ext_val_list.append(x_ext_val)
    y_ext_val_list.append(y_ext_val)

x_val = x_val_list[0]
x_ext_val = x_ext_val_list[0]
y_val = y_val_list[0]
y_ext_val = y_ext_val_list[0]
val_time_id = val_time_list[0]

for ii in range(1,len(val_storm_list)):
    x_val = np.concatenate((x_val,x_val_list[ii]),axis=0)
    x_ext_val = np.concatenate((x_ext_val,x_ext_val_list[ii]),axis=0)
    y_val = np.concatenate((y_val,y_val_list[ii]),axis=0)
    y_ext_val = np.concatenate((y_ext_val,y_ext_val_list[ii]),axis=0)
    val_time_id = np.concatenate((val_time_id,val_time_list[ii]),axis=0)
    
val_time = timei[val_time_id]
df_x_val = np.asarray(x_val).reshape(-1,SEQUENCE_SIZE,len(xcol))
x_ext_val = np.asarray(x_ext_val).reshape(-1,SEQUENCE_SIZE,len(xcols))
df_y_val = np.asarray(y_val).reshape(-1,1)    
print("Shape of x validation set: {}".format(df_x_val.shape))
print("Shape of y val set: {}".format(df_y_val.shape))
 
x_test_list = []
y_test_list = []
test_time_list = [] 

x_ext_test_list = []
y_ext_test_list = []

for ii in range(len(test_storm_list)):
    print("第{}个测试磁暴".format(ii))
    ymd2num0 = mdate.date2num(datetime.strptime(test_storm_list[ii][0],'%Y%m%d'))
    ymd2num1 = mdate.date2num(datetime.strptime(test_storm_list[ii][1],'%Y%m%d'))
    idx = (t_in >= ymd2num0)&(t_in <ymd2num1)
    omni_test = omni_in[idx]
    dst_test = dst_lag[idx]
   
    x_omni_test,y_dst_test,test_idx = to_sequences_MD_lag(SEQUENCE_SIZE,omni_test, dst_test,xcols = xcol)
    x_test_list.append(x_omni_test)
    y_test_list.append(y_dst_test)
    test_time_list.append(test_idx)
    
    x_ext_test,y_ext_test,_ = to_sequences_MD_lag(SEQUENCE_SIZE,omni_test, dst_test,xcols = xcols)
    x_ext_test_list.append(x_ext_test)
    y_ext_test_list.append(y_ext_test)
    

x_test = x_test_list[0]
x_ext_test = x_ext_test_list[0]
y_test = y_test_list[0]
y_ext_test = y_ext_test_list[0]

test_time_id = test_time_list[0]

for ii in range(1,len(test_storm_list)):
    x_test = np.concatenate((x_test,x_test_list[ii]),axis=0)
    x_ext_test = np.concatenate((x_ext_test,x_ext_test_list[ii]),axis=0)
    y_test = np.concatenate((y_test,y_test_list[ii]),axis=0)
    y_ext_test = np.concatenate((y_ext_test,y_ext_test_list[ii]),axis=0)
    test_time_id = np.concatenate((test_time_id,test_time_list[ii]),axis=0)
 

test_time = timei[test_time_id]
df_x_test = np.asarray(x_test).reshape(-1,SEQUENCE_SIZE,len(xcol))
x_ext_test = np.asarray(x_ext_test).reshape(-1,SEQUENCE_SIZE,len(xcols))
df_y_test = np.asarray(y_test).reshape(-1,1)


X = np.concatenate([df_x_train,df_x_val,df_x_test],axis=0)
y = np.concatenate([df_y_train,df_y_val,df_y_test],axis=0)

Ext_features = np.concatenate([x_ext_train,x_ext_val,x_ext_test],axis=0)

#scipy.io.savemat('test_storm_arr01.mat',{'test_time':test_time,'x_test':x_test,'x_ext_test':x_ext_test,\
#                 'y_test':y_test})

metrics = ['r2_score', 'mean_absolute_error', 'root_mean_squared_error', 'rmse_over_stdev']    
p = 0.05
dropout_rate = 0.1

inputs_ae = Input(shape=(SEQUENCE_SIZE, 1))
encoded_ae1 = Bidirectional(LSTM(20, return_sequences=True, dropout=dropout_rate))(inputs_ae, training=True)
encoded_ae = LSTM(15, return_sequences=True, dropout=dropout_rate)(encoded_ae1, training=True)

decoded_ae1 = LSTM(15, return_sequences=True, dropout=dropout_rate)(encoded_ae, training=True)
decoded_ae =Bidirectional(LSTM(20, return_sequences=True, dropout=dropout_rate))(decoded_ae1, training=True)
out_ae = TimeDistributed(Dense(1))(decoded_ae)

sequence_autoencoder = Model(inputs_ae, out_ae)
sequence_autoencoder.compile(optimizer='adam', loss='huber_loss', metrics='mse')
reduce_lr1 = tf.keras.callbacks.ReduceLROnPlateau()
mchpt1 = tf.keras.callbacks.ModelCheckpoint(
filepath='.\\storm_result\\autocoder0'+str(laghead)+'.ckpt',
save_best_only=False,
save_weights_only=True)

early_stopping =tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
lstm_h = sequence_autoencoder.fit(df_x_train, df_y_train, validation_data = (df_x_val,df_y_val), batch_size=64, epochs=300, verbose=2, shuffle = True,callbacks=[reduce_lr1,mchpt1,early_stopping])
#sequence_autoencoder.save_weights(model_weights_path1)
#autocoder_hist = autocoder_hist.append(lstm_h.history,ignore_index=True)
#autocoder_best_score.append(mchpt1.best)

encoder = Model(inputs_ae, encoded_ae)
XX = encoder.predict(X)
XXE = np.concatenate([XX, Ext_features], axis=2)
XXE.shape      
        
### SPLIT TRAIN TEST ###
X_train,  X_test = XXE[:len(df_x_train)], XXE[-len(df_x_test):]
y_train,  y_test = y[:len(df_y_train)], y[-len(df_y_test):]
        
X_val = XXE[len(df_x_train):len(df_x_train)+len(df_x_val)]
y_val = y[len(df_y_train):len(df_y_train)+len(df_y_val)]
### SCALE DATA ##
scaler1 = StandardScaler()
X_train = scaler1.fit_transform(X_train.reshape(-1,XXE.shape[-1])).reshape(-1,SEQUENCE_SIZE,XXE.shape[-1])
X_test = scaler1.transform(X_test.reshape(-1,XXE.shape[-1])).reshape(-1,SEQUENCE_SIZE,XXE.shape[-1])
X_val = scaler1.transform(X_val.reshape(-1,XXE.shape[-1])).reshape(-1,SEQUENCE_SIZE,XXE.shape[-1])
tf.random.set_seed(1)
### DEFINE FORECASTER ###
reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau()
mchpt2 = tf.keras.callbacks.ModelCheckpoint(
filepath='.\\storm_result\\forecaster0'+str(laghead)+'.ckpt',
save_best_only=False,
save_weights_only=True)
        
inputs1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
conv1 = Conv1D(filters=8, kernel_size=1)(inputs1,training=True)
Dropout(dropout_rate)(conv1)
attention_mul = attention_3d_block2(conv1)
lstm1 = Bidirectional(LSTM(20, return_sequences=True, dropout=dropout_rate))(attention_mul, training=True)

lstm1 = LSTM(15, return_sequences=True, dropout=dropout_rate)(lstm1)
conv1 = Conv1D(filters=8, kernel_size=1)(lstm1,training=True)
Dropout(dropout_rate)(conv1)
conv1 = Flatten()(conv1)
dense1 = Dense(10)(conv1)
out1 = Dense(1)(dense1)

model1 = Model(inputs1, out1)
model1.compile(loss='huber_loss', optimizer='adam', metrics='mse')        

### FIT FORECASTER ### validation_split=0.1 #validation_data=(X_val, y_val)
history = model1.fit(X_train, y_train, validation_data = (X_val,y_val),epochs=150, batch_size=64, verbose=2, shuffle = True, callbacks=[reduce_lr2,mchpt2,early_stopping])        
        
def test_stoc_drop1(df_x_test,ext_x_test,enc, NN): ##for model 1
     
         
     X,  F = df_x_test,ext_x_test
     
     enc_pred = np.vstack(enc([X]))
     enc_pred = np.concatenate([enc_pred, F], axis=2)
     trans_pred = scaler1.transform(enc_pred.reshape(-1,enc_pred.shape[-1])).reshape(-1,SEQUENCE_SIZE,enc_pred.shape[-1])
     NN_pred = NN([trans_pred])
     
     return np.vstack(NN_pred)       
        
def holdout_uncertainty_model11(df_x_test,ext_x_test,df_y_test,p):
    
    pred1 = []
   
   #  r_test = []
    enc = K.function([encoder.layers[0].input], [encoder.layers[-1].output])
    NN = K.function([model1.layers[0].input], [model1.layers[-1].output])

    for i in tqdm.tqdm(range(0,100)):
        
        pred1_test = test_stoc_drop1(df_x_test,ext_x_test, enc = enc, NN = NN)
                             
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
        
err_rmse = []
err_mae = []
err_r2 = []
err_median = []
err_rmse_std  = []        
result_yhat = []
result_ytest = []
result_pp = []      
tt_list = []  
for kk in range(0,len(test_storm_list)):
    ymd2num0 = mdate.date2num(datetime.strptime(test_storm_list[kk][0],'%Y%m%d'))
    ymd2num1 = mdate.date2num(datetime.strptime(test_storm_list[kk][1],'%Y%m%d'))
    fp  = (test_time >= ymd2num0)&(test_time <ymd2num1)
    x_fp = df_x_test[fp]
    x_ext_fp = x_ext_test[fp]
    y_fp = df_y_test[fp]
    tt = test_time[fp]
    tt_list.append(tt[:-laghead])
    holdout_uncer11, pp = holdout_uncertainty_model11(x_fp,x_ext_fp,y_fp,p)
    
    act = holdout_uncer11['true'].values
    yact = act[:-laghead]
    pred = holdout_uncer11['medium'].values
    ypred = pred[laghead:]
    result_yhat.append(pred)
    result_ytest.append(act)
    result_pp.append(pp[laghead:])
    rmse = np.sqrt(mean_squared_error(yact,ypred))
    mae = mean_absolute_error(yact,ypred)
    r2 = r2_score(yact,ypred)
    mdn = median_absolute_error(yact, ypred)
    rmse_over_std = rmse/np.std(yact)
    print('='*30)
    print(rmse)
    err_rmse.append(rmse)
    err_mae.append(mae)
    err_r2.append(r2)
    err_median.append(mdn)
    err_rmse_std.append(rmse_over_std)        

data = {'result_yhat':result_yhat,'result_yact':result_ytest,'result_pp':result_pp,\
        'err_rmse':err_rmse,'err_r2':err_r2,'tt_list':tt_list}            
import pickle
with open(r'./output01'+str(laghead)+'.pkl','wb') as f:
    pickle.dump(data,f)       


stdevs = np.std(pp,axis=1)
residuals = yact - ypred       

predicted_pi = np.linspace(0, 1, 100)
observed_pi = [calculate_density(residuals,stdevs,quantile)
               for quantile in tqdm_notebook(predicted_pi, desc='Calibration')]

calibration_error = ((predicted_pi - observed_pi)**2).sum()
print('Calibration error = %.2f' % calibration_error)           
#with open(r'./output0'+str(laghead)+'pkl','rb') as f:
#      newdata = pickle.load(f)    
#plt.legend(('True','True shift','Predicted'))

#plt.title("LSTM + Autoender at storm1 ")

#plotcorr(yact, ypred)

### PLOT AVG AND UNCERTAINTY OF RESULTS ###
#bar = plt.bar([0,1], [np.mean(mmae11), np.mean(mmae12)], 
#              yerr=[2.95*np.std(mmae11), 2.95*np.std(mmae12)])
#plt.xticks([0,1], ['model1','model2'], rotation=90)
#bar[0].set_color('cyan'), bar[1].set_color('magenta')

#https://wdc.kugi.kyoto-u.ac.jp/dstdir/index.html


