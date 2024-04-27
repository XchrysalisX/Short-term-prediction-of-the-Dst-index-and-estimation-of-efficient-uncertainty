# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 19:46:27 2024

@author: huang
"""
import numpy as np
import matplotlib.pyplot as plt
font = {'family':'Times New Roman','weight':'normal','size':12}

fig,ax = plt.subplots(2,2,figsize=(8,8))
MA = [0.04178,0.01059,0.02819,0.05433,0.07959,0.10273,0.12380,0.14289,0.16019]
RMSCE = [0.04734,0.01574,0.03209,0.06145,0.08897,0.11414,0.1371,0.15804,0.17712]
ax[0,0].plot(np.arange(1.6,3.4,0.2),MA,'k-o',label = 'Miscalibration Area')
ax[0,0].plot(np.arange(1.6,3.4,0.2),RMSCE,'b-x',label = 'RMSCE')
ax[0,0].set_ylim([0,0.18])
ax[0,0].set_yticks(np.arange(0,0.18,0.05))
ax[0,0].set_yticklabels([0,0.05,0.10,0.15],fontproperties = font)
ax[0,0].set_xlim([1.5,3.3])
ax[0,0].set_xticks(np.arange(1.6,3.4,0.2))
ax[0,0].set_xticklabels([1.6,1.8,2.0,2.2,2.4,2.6,2.8,3,3.2],fontproperties = font)
ax[0,0].legend()
#ax[0,0].set_xlabel('\u03B1',fontproperties = font)
#ax[0,0].set_ylabel('Uncertainty Metric',fontproperties = font)
ax[0,0].set_title('1-h ahead',font)

MA =[0.09247,0.05659,0.02393,0.00740,0.03307,0.05773,0.08027,0.10079,0.11979]
RMSCE = [0.10178,0.06227,0.02655,0.00817,0.03671,0.06382,0.08862,0.1113,0.13226]
ax[0,1].plot(np.arange(1.6,3.4,0.2),MA,'k-o',label = 'Miscalibration Area')
ax[0,1].plot(np.arange(1.6,3.4,0.2),RMSCE,'b-x',label = 'RMSCE')
ax[0,1].set_ylim([0,0.17])
ax[0,1].set_yticks(np.arange(0,0.17,0.05))
ax[0,1].set_yticklabels([0,0.05,0.10,0.15],fontproperties = font)
ax[0,1].set_xlim([1.5,3.3])
ax[0,1].set_xticks(np.arange(1.6,3.4,0.2))
ax[0,1].set_xticklabels([1.6,1.8,2.0,2.2,2.4,2.6,2.8,3,3.2],fontproperties = font)
ax[0,1].set_title('2-h ahead',font)
#ax[0,1].legend()
#ax[0,1].set_xlabel('\u03B1',fontproperties = font)
#ax[0,1].set_ylabel('Uncertainty Metric',fontproperties = font)

MA = [0.13245,0.09747,0.06527,0.03553,0.00828,0.01751,0.04045,0.06191,0.08173]
RMSCE = [0.14628,0.10737,0.07180,0.03910,0.00938,0.01948,0.04479,0.06841,0.09023]
ax[1,0].plot(np.arange(1.6,3.4,0.2),MA,'k-o',label = 'Miscalibration Area')
ax[1,0].plot(np.arange(1.6,3.4,0.2),RMSCE,'b-x',label = 'RMSCE')
ax[1,0].set_ylim([0,0.17])
ax[1,0].set_yticks(np.arange(0,0.17,0.05))
ax[1,0].set_yticklabels([0,0.05,0.10,0.15],fontproperties = font)
ax[1,0].set_xlim([1.5,3.3])
ax[1,0].set_xticks(np.arange(1.6,3.4,0.2))
ax[1,0].set_xticklabels([1.6,1.8,2.0,2.2,2.4,2.6,2.8,3,3.2],fontproperties = font)
#ax[1,0].legend()
ax[1,0].set_xlabel('\u03B1',fontproperties = font)
ax[1,0].set_ylabel('Uncertainty Metric',fontproperties = font)
ax[1,0].set_title('3-h ahead',font)

MA = [0.16106,0.12715,0.0953,0.06581,0.03839,0.01309,0.01111,0.03249,0.05260]
RMSCE = [0.17759,0.13963,0.10431,0.07167,0.04159,0.01451,0.01413,0.03734,0.05944] 
ax[1,1].plot(np.arange(1.6,3.4,0.2),MA,'k-o',label = 'Miscalibration Area')
ax[1,1].plot(np.arange(1.6,3.4,0.2),RMSCE,'b-x',label = 'RMSCE')
ax[1,1].set_ylim([0,0.18])
ax[1,1].set_yticks(np.arange(0,0.17,0.05))
ax[1,1].set_yticklabels([0,0.05,0.10,0.15],fontproperties = font)
ax[1,1].set_xlim([1.5,3.3])
ax[1,1].set_xticks(np.arange(1.6,3.4,0.2))
ax[1,1].set_xticklabels([1.6,1.8,2.0,2.2,2.4,2.6,2.8,3,3.2],fontproperties = font)
#ax[1,1].legend()
#ax[1,1].set_xlabel('\u03B1',fontproperties = font)
#ax[1,1].set_ylabel('Uncertainty Metric',fontproperties = font)
ax[1,1].set_title('4-h ahead',font)