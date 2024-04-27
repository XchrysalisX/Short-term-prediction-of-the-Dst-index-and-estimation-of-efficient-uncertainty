# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:52:39 2024

@author: huang
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from recalibration import plot_recalibration

num_stds_confidence_bound = [1.8,2.2,2.4,2.8]  #-1

#num_stds_confidence_bound = [2,2.5,3,3]
font = {'family':'Times New Roman','weight':'normal','size':12}
storm_list = [3]
lag_list = np.arange(1,5).reshape(2,2)

for k in range(len(storm_list)):
    
    for ii in range(2):
        fig, ax = plt.subplots(2,2,figsize=(8,6))
        for jj in range(2):
            
            with open(r'E:\tfd\BayLSTM/output0'+str(lag_list[ii,jj])+'.pkl','rb') as f:
                  newdata = pickle.load(f)   
            result_yhat = newdata['result_yhat']
            result_yact = newdata['result_yact']
            result_pp = newdata['result_pp']
            tt = newdata['tt_list']
            
            storm_yhat = result_yhat[storm_list[k]][lag_list[ii,jj]:]
            storm_yact = result_yact[storm_list[k]][:-lag_list[ii,jj]]
            storm_tt = tt[storm_list[k]]
            storm_pp = result_pp[storm_list[k]]
            mnpp = np.mean(storm_pp,axis=1)
            repmn = np.tile(mnpp,[100,1]).T
            ep = np.std(storm_pp,axis=1)
            stdevs = ep
            intervals = num_stds_confidence_bound[2*ii+jj] * np.squeeze(stdevs)
            intervals_lower_upper = [storm_yhat - intervals, storm_yhat + intervals]
            lims_ext = [
                int(np.floor(np.min(intervals_lower_upper[0]))),
                int(np.ceil(np.max(intervals_lower_upper[1]))),
            ]
            ax[0,jj].plot_date(storm_tt,storm_yact,color = 'black',\
                        linestyle='solid',linewidth=2,markersize = 1)
            ax[0,jj].plot_date(storm_tt,storm_yhat,color = '#FF0000',\
                        linestyle='dotted',linewidth=2,markersize = 2)
            ax[0,jj].fill_between(storm_tt,intervals_lower_upper[0],intervals_lower_upper[1],alpha=0.2,edgecolor='#FF0000',facecolor='#FF0000',
                               linewidth=2, linestyle='solid', antialiased=True)
            ax[0,jj].set_xticks(np.linspace(storm_tt[0],storm_tt[-1],5))
            ax[0,jj].set_xticklabels([])
            ax[0,jj].set_yticks(np.arange(-200,30,50))
            ax[0,jj].set_yticklabels(np.arange(-200,30,50),fontproperties = font)
            ax[0,jj].set_title(str(lag_list[ii,jj])+'-h ahead',font)
    #       ax[0].set_yticklabels(np.arange(-350,50,100),font)
    #        ax[0,jj].legend(('True Dst','Predicted Dst','Predicted interval'), prop = font,loc='lower right')
            ax[0,jj].set_ylabel('Dst(nT)',font)
            
            exce_dw =  intervals_lower_upper[0] - storm_yact
            exce_dw[exce_dw<0] = 0
            
            exce_up = storm_yact - intervals_lower_upper[1]
            exce_up[exce_up<0] = 0
            
            exce = np.vstack([exce_up,exce_dw]).T
            exce_df = pd.DataFrame(exce)
            exce_df.columns = ['Exceeding upper','Exceeding lower']
            exce_df.index = storm_tt
            
            exce_df.plot.area(stacked = False,alpha = 0.2,ax=ax[1,jj])
            ax[1,jj].set_xticks(np.linspace(storm_tt[0],storm_tt[-1],5))
            ax[1,jj].set_yticks(np.arange(0,22,5))
            ax[1,jj].set_yticklabels(np.arange(0,22,5),fontproperties = font)
            ax[1,jj].set_xticklabels([])
            ax[1,jj].set_xticklabels(['22/08/2018','25/08/2018','28/08/2018','31/08/2018','02/09/2018'],\
                         rotation = 30,fontproperties = font)
            ax[1,jj].set_ylabel('Dst(nT)',font)

      
     