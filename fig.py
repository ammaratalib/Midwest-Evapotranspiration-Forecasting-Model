# -*- coding: utf-8 -*-



# fall 2020
#  ANAT&PHY 335 — PHYSIOLOGY  5 credits
#or
#ANAT&PHY 337 — HUMAN ANATOMY   ANAT&PHY 338 — HUMAN ANATOMY LABORATORY   total 5 credits (fall)

#spring 2020
#Genetics 466: Principles of Genetics 3 credits   fall spring summer 
#Genetics 545: Genetics Laboratory 2 credit lab  # fall

# physiology 335   5 credits not lab

"""
Created on Fri Aug 23 16:59:37 2019

@author: Ammara


"""


###### calibration location 


### testing location
######



from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import numpy as np
from math import sqrt
import matplotlib.dates as mdates

from sklearn.metrics import r2_score

import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend



#pbias
#y_test.index=np.arange(0,len(y_test))
#sse=mean_squared_error(df.rfL1ET2,df.ET2_obs)
#var = np.var(df.rfL1ET2)
#bias = sse -var-0.01

#pbias=((np.sum(df.ET2_obs-df.rfL1ET2))*100)/np.sum(df.ET2_obs

import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
from scipy.stats.stats import pearsonr

def NSE(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
#    s,o = filter_nan(s,o)
    return 1 - ((sum((s-o)**2))/(sum((o-np.mean(o))**2)))

############################################################################################################
def soil_df(df):
    if (df['ID'] =='IA_Br3'):
        return 'clay loam'
    elif (df['ID'] =='IA_Br1'):
        return 'loam'
    elif (df['ID'] =='WI_CS') or (df['ID'] =='MI_irri'):
        return 'loamy sand'
    elif (df['ID'] =='MI_T4') or (df['ID'] =='MI_nonirrig')or (df['ID'] =='MI_T3'):
        return 'sandy loam'
    elif (df['ID'] =='IL_B02'):
        return 'silty clay'
    elif (df['ID'] =='NB_Ne1'):
        return 'silty clay loam'
    else: 
        return 'silt loam'      
#########################################################################################################
        
def state_df(df):
    if (df['ID'] =='MI_T3')or (df['ID'] =='MI_T4')or (df['ID'] =='MI_irrig')or (df['ID'] =='MI_nonirri'):
        return 'MI'
    elif (df['ID'] =='NB_Ne2')or (df['ID'] =='NB_Ne1') or (df['ID'] =='NB_Ne3'):
        return 'NE'
    elif (df['ID'] =='MN_Ro3') or (df['ID'] =='MN_Ro1')or (df['ID'] =='MN_Ro5') or (df['ID'] =='MN_Ro2') or (df['ID'] =='MN_Ro6'):
        return 'MN'
    elif (df['ID'] =='IL_Bo1') or (df['ID'] =='IL_1B1') or (df['ID'] =='IL_B02'):
        return 'IL'
    elif (df['ID'] =='IA_Br1')or (df['ID'] =='IA_Br3'):
        return 'IA'
    elif (df['ID'] =='WI_CS'):
        return 'WI'
    else: 
        return 'OH'  
#######################################################################################################
  
def irrig_non(df):
    if (df['ID'] =='NB_Ne2')or (df['ID'] =='MI_irrig')or (df['ID'] =='WI_CS')or (df['ID'] =='NB_Ne1'):
        return 'irrigated'
    else: 
        return 'non-irrigated'  

#############################################################################################################

def color(df):
    if (df['soil'] =='silty clay'):
        return 'violet'
    if (df['soil'] =='silty clay loam'):
        return 'indigo'
    if (df['soil'] =='clay loam'):
        return 'blue'
    if (df['soil'] =='silt loam'):
        return 'green'
    if (df['soil'] =='loam'):
        return 'yellow'   
    if (df['soil'] =='sandy loam'):
        return 'orange'   
    else:
        'red'
        return
####################################################################################################
def month_n(df):
    if (df['month'] ==4):
        return 'Apr'
    if (df['month'] ==5):
        return 'May'
    if (df['month'] ==6):
        return 'Jun'
    if (df['month'] ==7):
        return 'Jul'
    if (df['month'] ==8):
        return 'Aug'   
    if (df['month'] ==9):
        return 'Sep' 
    if (df['month'] ==10):
        return 'Oct' 
    else:
        'Nov'
        return
###########################################################################################################
def marker(df):
    if (df['soil'] =='silty clay'):
        return '+'
    if (df['soil'] =='silty clay loam'):
        return ':'
    if (df['soil'] =='clay loam'):
        return 'v'
    if (df['soil'] =='silt loam'):
        return 's'
    if (df['soil'] =='loam'):
        return 'p'   
    if (df['soil'] =='sandy loam'):
        return 'o'   
    else:
        '^'
        return
###########################################################################################################  
def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic

#marker='silty clay':"+",'silty clay loam':':','clay loam':'v','silt loam':'s','loam':'s','sandy loam':'o'

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
corn = df[(df['crop']=='corn')]
soy = df[(df['crop']=='soy')]
potat=df.loc[df['crop'].isin(['potat1','potat2'])]

######################################################################################################################

np.mean(np.abs((y_true - y_pred) / y_true)) * 100
###worked
def get_legend_markers(D_label_color, marker='+', marker_kws={"linestyle":""}):
    markers = [plt.Line2D([0,0],[0,0],color=color, marker=marker, **marker_kws) for color in D_label_color.values()]
    return (markers, D_label_color.keys())    
#classes = ['silty clay loam', 'silt loam', 'silty clay', 'sandy loam','clay loam']
D_label_color = {"silty clay loam":"indigo", "silt loam":"green", "silty clay":"violet","sandy loam":"orange","clay loam":"blue"}
marker = {"silty clay loam":"+", "silt loam":"v", "silty clay":"s","sandy loam":"p","clay loam":"o"}
#marker='silty clay':"+",'silty clay loam':':','clay loam':'v','silt loam':'s','loam':'s','sandy loam':'o'
#markers=[":","s","v","o","v"]
ax=sns.regplot(data=df, x="ET1_obs", y="rfL1ET1",fit_reg=False,scatter_kws={'facecolors':df['color']})
ax.legend(*get_legend_markers(D_label_color))
plt.show()
ax.set_ylabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.set_ylim(0,10)
ax.set_xlim(0,10)
plt.show()

markers = ['o', 's', 'p', 'x', '^']
ax=sns.lmplot(data=df, x="ET1_obs", y="rfL1ET1", 
                   hue='soil', markers=markers, scatter=True, fit_reg=False)
ax.legend()
plt.show()

###########################################################################################################3

#############################################################################################################
#sns.plt.show()

silty clay	        r	v
silty clay loam 	o	.
clay loam	        y	p
silt loam	        g	+
loam	            b	o
sandy loam	        i	s
loamy sand	        v	u

# corn 6 10
##########################################################################################################
## Figure 1
########################################################################################################3333
#corn random forest

#########################################################################################################
df=corn
df.soil.unique()
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
y1=(df["test_obs16"][(df['soil']=='sandy loam')])
x1= (df["rftest_pred16"][(df['soil']=='sandy loam')])
y2=(df["test_obs16"][(df['soil']=='silt loam')])
x2= (df["rftest_pred16"][(df['soil']=='silt loam')])
y3=(df["test_obs16"][(df['soil']=='loam')])
x3=(df["rftest_pred16"][(df['soil']=='loam')])
# width height
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(x1,y1,'.', color="blue",markersize=2,label="Sandy Loam")  ##0.65
ax.plot(x2,y2,'.', color="blue",markersize=1,label="Silt Loam")###0.70
ax.plot(x3,y3,'.', color="blue",markersize=0.5,label="Loam")###0.75
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("Corn",size=10)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,12)
ax.set_xlim(0,12)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


r2_score(obs,pre)

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))


ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias






from math import log

def AIC( obs,pre):
            
    num_params=16+2
    mse=mean_squared_error(obs, pre)
    return calculate_aic(len(obs), mse, num_params)

def AIC( obs,pre):
            
    num_params=5+2
    mse=mean_squared_error(obs, pre)
    return calculate_aic(len(obs), mse, num_params)

########################################################################################
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
corn = df[(df['crop']=='corn')]
soy = df[(df['crop']=='soy')]
potat=df.loc[df['crop'].isin(['potat1','potat2'])]

#df=pd.read_csv('pre_train.csv')

df.index = pd.DatetimeIndex(df.TIMESTAMP)
back=df
df=df.rename_axis('TIMESTAMPf').reset_index()

#mask = (df['irr_nonirr']=="irrigated")
mask = (df['irr_nonirr']=="non-irrigated")

df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
#irr=df
nonirr=df


df=nonirr


  
#mask = (df['TIMESTAMPf'] >'2003-4-1') & (df['TIMESTAMPf'] <= '2003-10-31')

#mask = (df['TIMESTAMPf'] >'2004-4-1') & (df['TIMESTAMPf'] <= '2004-10-31')

# end date will be one day more than your desired end day

#'rnn16_L1_25', 'rnntest_pred16', 'rnn16_L1_100', 'rnn16_L1_200

obs=df.test_obs16
#pre=df.rnn16_L1_25
#pre=df.rnntest_pred16
#pre=df.rnn16_L1_100
#pre=df.rnn16_L1_200

#pre=df["vp_prec"]
#pre=df["evi_crop_coeff"]
#pre=df["solar_cumGDD"]
#pre=df["remove_8"]
pre=df["solar_SW"]




r2_score(obs,pre)

NSE(pre,obs)

will(pre,obs)

mean_absolute_error(obs,pre)

ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias


pearsonr(obs,pre)

#pre=df.CNNtest_pred16
#pre=df.rftest_pred16
#pre=df.rnntest_pred16

pre=df.CNNtest_pred5








def AIC( obs,pre):
            
    num_params=5+2
    mse=mean_squared_error(obs, pre)
    return calculate_aic(len(obs), mse, num_params)

AIC(obs,pre)

# RF_16, LSTM_16, CNN_16, RF_11, LSTM_11, CNN_11, RF_5, LSTM_5, CNN_5

s=[-2155.8,-879.95,1336.15,-2018.83,-584.96,-435.5,-421.6,1384.5,1593.32]


ans=(s-min(s))/(max(s)-min(s))
ans

obs=df.test_obs16
#pre=df.rftest_pred16
pre=df.rnntest_pred16

NSE(pre,obs)

will(pre,obs)


#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
r2_score(obs,pre)

NSE(pre,obs)
will(pre,obs)

ubRMSE(obs,pre)
RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

mean_absolute_error(obs,pre)

pearsonr(obs,pre)


import hydroeval
##########################################################################################################
#corn LSTM
#############################################################################################################
#df=corn
df=corn

df.soil.unique()
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
y1=(df["test_obs16"][(df['soil']=='sandy loam')])
x1= (df["rnntest_pred16"][(df['soil']=='sandy loam')])

y2=(df["test_obs16"][(df['soil']=='silt loam')])
x2= (df["rnntest_pred16"][(df['soil']=='silt loam')])
y3=(df["test_obs16"][(df['soil']=='loam')])
x3=(df["rnntest_pred16"][(df['soil']=='loam')])

# width height
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(x1,y1,'.', color="blue",markersize=2,label="Sandy Loam")
ax.plot(x2,y2,'.', color="blue",markersize=1,label="Silt Loam")
ax.plot(x3,y3,'.', color="blue",markersize=0.5,label="Loam")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("Corn",size=10)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,12)
ax.set_xlim(0,12)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


obs=df.test_obs16
pre=df.NOAA

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)

r2_score(obs,pre)

NSE(pre,obs)
will(pre,obs)

mean_absolute_error(obs,pre)

ubRMSE(obs,pre)
RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias


pearsonr(obs,pre)

#########################################################################################################

##soybean random forest
##########################################################################################################################
df=soy
df.soil.unique()
y2=(df["test_obs16"][(df['soil']=='silt loam')])
x2= (df["rftest_pred16"][(df['soil']=='silt loam')])
y3=df["test_obs16"][(df['soil']=='loam')]
x3= df["rftest_pred16"][(df['soil']=='loam')]



plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
ax.plot(x2,y2,'.', color="blue",markersize=1,label="Silt Loam")
ax.plot(x3,y3,'.', color="blue",markersize=0.5,label="Loam")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.locator_params(axis='y', nbins=5)
ax.locator_params(axis='x', nbins=5)

line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("Soybeans",size=10)
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,10)
ax.set_xlim(0,10)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#obs=df.test_obs16
#pre=df.rftest_pred16

r2_score(obs,pre)

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias



#######################################################################################################
df=soy
df.soil.unique()
y2=(df["test_obs16"][(df['soil']=='silt loam')])
x2= (df["rnntest_pred16"][(df['soil']=='silt loam')])
y3=df["test_obs16"][(df['soil']=='loam')]
x3= df["rnntest_pred16"][(df['soil']=='loam')]

plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
ax.plot(x2,y2,'.', color="blue",markersize=1,label="Silt Loam")
ax.plot(x3,y3,'.', color="blue",markersize=0.5,label="Loam")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.locator_params(axis='y', nbins=5)
ax.locator_params(axis='x', nbins=5)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("Soybeans",size=10)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,10)
ax.set_xlim(0,10)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

obs=df.test_obs16
pre=df.rnntest_pred16

r2_score(obs,pre)

NSE(pre,obs)
will(pre,obs)

ubRMSE(obs,pre)
RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

mean_absolute_error(obs,pre)

pearsonr(obs,pre)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias


##########################################################################################################
##potato random forest
#####################################################################################################
df=potat
df.soil.unique()
#array(['silt loam', 'loam'], dtype=object) 
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()

y4=df["test_obs16"][(df['soil']=='loamy sand')]
x4= df["rftest_pred16"][(df['soil']=='loamy sand')]

ax.plot(x4,y4,'.', color="blue",markersize=2,label='Loamy Sand')
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)

line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("Potatoes",size=10)
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,6)
ax.set_xlim(0,6)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
obs=df.test_obs16
pre=df.rftest_pred16





r2_score(obs,pre)
NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias



#########################################################################################################

##########################################################################################################
##potato lstm
#####################################################################################################
df=potat
df.soil.unique()
#array(['silt loam', 'loam'], dtype=object) 
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()

y4=df["test_obs16"][(df['soil']=='loamy sand')]
x4= df["rnntest_pred16"][(df['soil']=='loamy sand')]

ax.plot(x4,y4,'.', color="blue",markersize=2,label='Loamy Sand')
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)

line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("Potatoes",size=10)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,6)
ax.set_xlim(0,6)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

obs=df.test_obs16
pre=df.rnntest_pred16

r2_score(obs,pre)

NSE(pre,obs)
will(pre,obs)

ubRMSE(obs,pre)
RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

mean_absolute_error(obs,pre)

pearsonr(obs,pre)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias


#limits for forcast
#260,250,140
################################################################################################
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
#df_MI=(df[(df['state']=='MI')])
#df_NE=(df[(df['state']=='NE')])
#df_MN=(df[(df['state']=='MN')])
#df_IA=(df[(df['state']=='IA')])
df_WI=(df[(df['state']=='WI')])

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

#df=df_MI
#df=df_NE
#df=df_MN
#df=df_IA
df=df_WI

#array(['MI_T3', 'NB_Ne2', 'MN_Ro3', 'IA_Br1', 'WI_CS', 'MN_Ro1'],
 #   irrigated  WI_CS, NE2
   
obs_ecdf=ecdf((df.test_obs16))
rf_ecdf_16=ecdf((df.rftest_pred16))
rnn_ecdf_16=ecdf((df.rnntest_pred16))

plt.rcParams['figure.figsize'] = (1.5,1)
fig, ax = plt.subplots()
ax.plot(obs_ecdf[0],obs_ecdf[1],c='black',label="observed",linestyle='--')
ax.plot(rf_ecdf_16[0],rf_ecdf_16[1],c='blue',label="RF_16",linestyle='--')
ax.plot(rnn_ecdf_16[0],rnn_ecdf_16[1],c='green',label="LSTM_16",linestyle='--')

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.1)
ax.legend(loc='lower right',prop={'size': 4})
ax.set_title("Empirical CDF",size=5)
ax.set_xlabel('ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('ECDF')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,1)
#MI 200, NE 300, MN 300, IW 250,WI 250
ax.set_xlim(0,6)  #IA

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()



##########################################################################################################
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
#df_MI=(df[(df['state']=='MI')])
#df_NE=(df[(df['state']=='NE')])
#df_MN=(df[(df['state']=='MN')])
#df_IA=(df[(df['state']=='IA')])
df_WI=(df[(df['state']=='WI')])

## figure 3  part b
#df=df_MI
#df=df_NE
#df=df_MN
#df=df_IA
df=df_WI

#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
y1=(df["test_obs16"])
x1= (df["rftest_pred16"])
y2=(df["test_obs16"])
x2= (df["rnntest_pred16"])

# width height
plt.rcParams['figure.figsize'] = (1.5,1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(x1,y1,'o', color="blue",markersize=0.3,label="RF_16")
ax.plot(x2,y2,'o', color="green",markersize=0.3,label="LSTM_16")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=0.3)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper left',prop={'size': 4})
#ax.set_title("Scatter plot Observed versus Models",size=5)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')

##NE=0,250, MN 300
#MI= 8, NE=12,MN= 12,IW= 10, WI=8 
ax.set_ylim(0,6)
ax.set_xlim(0,6)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()



r2_score(y1,x1)
r2_score(y2,x2)
### already mutiply with 25.4
sqrt(mean_squared_error(y1,x1))
sqrt(mean_squared_error(y2,x2))

pbias=(np.sum(x1-y1)/np.sum(y1))*100
pbias

pbias=(np.sum(x2-y2)/np.sum(y2))*100
pbias

ubRMSE(y1,x1)
ubRMSE(y2,x2)

####################################################################################################
#import sys
#sys.setrecursionlimit(10000)
## figure C
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
#df_MI=(df[(df['state']=='MI')])
#df_NE=(df[(df['state']=='NE')])
#df_MN=(df[(df['state']=='MN')])
#df_IA=(df[(df['state']=='IA')])
df_WI=(df[(df['state']=='WI')])

#df=df_MI
#df=df_NE
#df=df_MN
#df=df_IA
df=df_WI
#MI 200, NE 300, MN 300, IW 250,WI 250

#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
x1=(df["test_obs16"])
y1= (df["rftest_pred16"])
x2=(df["test_obs16"])
y2= (df["rnntest_pred16"])
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
#df.index = df.index.map(str)
x=df["TIMESTAMP"]

plt.rcParams['figure.figsize'] = (4,1.1)
fig, ax = plt.subplots()

ax.plot(x,x1, color="black",linestyle='none',marker='o',label="Observed",markersize=0.3)
ax.plot(x,y1, color="blue",linestyle='none',marker='o',label="RF_16",markersize=0.3)
ax.plot(x,y2, color="green",linestyle='none',marker='o',label="LSTM_16",markersize=0.3)
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
legend=ax.legend(loc='upper left',prop={'size': 4})
#ax.set_title("Location:Michigan, EC Tower ID:US-KM1 Non-Irrigated ,Crop:Corn, Soil:Sandy Loam",size=5)
#ax.set_title("Location:Nebraska, EC Tower ID:US-Ne2 Irrigated ,Crop:Corn and Soybean, Soil:Silt Loam",size=5)
#ax.set_title("Location:Minnesota, EC Tower ID:US-R01, US-R03 Non-Irrigated,Crop:Corn and Soybean, Soil:Silt Loam",size=5)
#ax.set_title("Location:Iowa, EC Tower ID:US-Br1 Non-Irrigated,Crop:Corn and Soybean, Soil:Loam",size=5)
ax.set_title("Location:Wisconsin, EC Tower ID:US-CS1 and US-CS3 Irrigated,Crop:Potatoes, Soil:Loamy Sand",size=5)

ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1
ax.set_ylabel('ET (mm)')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 4,direction='in')
ax.set_ylim(0,6)  # MI=200, NE,250 #
#MI 200, NE 300, MN 300, IW 250,WI 250

## 300 for NE
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
#ax.set_xscale('custom')
#ax.set_xticks(xticks)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#fig.autofmt_xdate() 
ax = plt.gca()
#plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


def export_legend(legend, filename="legend.png", expand=[-1,-1,1,1]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=600, bbox_inches=bbox)

export_legend(legend)
plt.show()
#########################################################################################################
## figurel correlation

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('corr.csv')
        
import matplotlib.pyplot as plt
import pandas
import numpy
mov_data= ['prcp7','prcp15','prcp30','VP','Tmax','LW','SW' ,'wind','SolarZenith','Albedo','EVI' ,'Soil','Irr_nonirr','Crop_cover','Crop_coeff','CumGDD','ET']
#data = pd.read_csv(url, names=names)
correlations = df.corr()
# plot correlation matrix
fig = plt.figure()
plt.rcParams['figure.figsize'] = (2.9,2.8)
fig, ax = plt.subplots()
subplotsize=[8.,8.]
figuresize=[10.,10.]   
left = 0.5*(1.-subplotsize[0]/figuresize[0])
right = 1.-left
bottom = 0.5*(1.-subplotsize[1]/figuresize[1])
top = 1.-bottom
fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
cax = ax.matshow(correlations,vmin=-0.4, vmax=1)
cax = ax.matshow(correlations, vmin=-0.4, vmax=1)
cb=fig.colorbar(cax,shrink=0.8, ticks=[1,0.8,0.6,0.4,0.2,0,-.2,-.4])
cb.ax.tick_params(labelsize=10)
ticks = np.arange(0,17,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(mov_data)

ax.set_yticklabels(mov_data)
ax.set_xticklabels(mov_data, rotation=90)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
plt.show()       
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


 #Pearson correlation coefficient
############################################################################################

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('imp.csv')

import os
import numpy as np
import matplotlib.pyplot as plt
x=list(df["imp"])
y=list(df["index"])

import matplotlib.pyplot as plt
#y = ['Physics','Chemistry','Economics','Philosophy','Computer Science','Sociology','Biology','History','Overall']
#x = [0.0,33.333333333333336,50.0,50.0,50.0,54.54545454545455,58.333333333333336,100.0,51.851851851851855]
#alternativeYlabels = ['0.02', '0.023', '0.024', '0.028', '0.029', '0.03', '0.031', '0.033', '0.037','0.038','0.043','0.044','0.08','0.11','0.13','0.29']plt.rcdefaults()
fig = plt.figure()
plt.rcParams['figure.figsize'] = (2.9,2.8)
fig, ax = plt.subplots()

# Example data
people= ['EVI','SolarZenith','SW','CumGDD','prcp7','Crop_cover','Tmax','prcp30','Crop_coeff','wind','prcp15',
'Irr_nonirr','Albedo','Soil','LW','VP']

y_pos = np.arange(len(y))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, x, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importance')
ax.set_title('Predictors Importance (RF_16)',size=8)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_xlim(0,0.4)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

############################################################################################

###lstm_predictor importance
import os
import numpy as np
import matplotlib.pyplot as plt
y= ['All Predictors','-CumGDD,-SolarZenith','-Crop_coeff,-EVI','-VP,-prcp7','-SW,-SolarZenith'
]

#x=[0.65,0.61,0.58,0.57,0.50]   all
#z=[0.71,0.70,0.67,0.63,0.52]   # irrigtaed
z=[0.60,0.54,0.52,0.53,0.47]   # non-irrigtaed

#x=list(df["imp"])
y=y
import matplotlib.pyplot as plt
#y = ['Physics','Chemistry','Economics','Philosophy','Computer Science','Sociology','Biology','History','Overall']
#x = [0.0,33.333333333333336,50.0,50.0,50.0,54.54545454545455,58.333333333333336,100.0,51.851851851851855]
#alternativeYlabels = ['0.02', '0.023', '0.024', '0.028', '0.029', '0.03', '0.031', '0.033', '0.037','0.038','0.043','0.044','0.08','0.11','0.13','0.29']plt.rcdefaults()
fig = plt.figure()
plt.rcParams['figure.figsize'] = (2.9,2.8)
fig, ax = plt.subplots()

# Example data
people= ['All Predictors','-CumGDD,-SolarZenith','-Crop_coeff,-EVI','-VP,-prcp7','-SW,-SolarZenith'
]

y_pos = np.arange(len(y))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

#ax.barh(y_pos, x, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('NSE')
#ax.set_title('Predictors Importance (LSTM_16)',size=8)
#ax.set_title('Predictors Importance, Irrigated (LSTM_16)',size=8)
ax.set_title('Predictors Importance, Non-Irrigated (LSTM_16)',size=8)

#ax.barh(y_pos, z, align='center',color='green')
ax.barh(y_pos, z, align='center',color='black')

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_xlim(0,1)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

###############################################################################################
# lstm_irrigated , non irrigated 
mask = (df['irr_nonirr']==1)
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
irr=df
nonirr=df


df=irr





#############################################################################################
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('predi_imp_irr.csv')
#df=pd.read_csv('forcast_imp_irr.csv')

import os
import numpy as np
import matplotlib.pyplot as plt
x=list(df["irri"])
y=list(df["index"])
z=list(df["non"])
import matplotlib.pyplot as plt
#y = ['Physics','Chemistry','Economics','Philosophy','Computer Science','Sociology','Biology','History','Overall']
#x = [0.0,33.333333333333336,50.0,50.0,50.0,54.54545454545455,58.333333333333336,100.0,51.851851851851855]
#alternativeYlabels = ['0.02', '0.023', '0.024', '0.028', '0.029', '0.03', '0.031', '0.033', '0.037','0.038','0.043','0.044','0.08','0.11','0.13','0.29']plt.rcdefaults()
fig = plt.figure()
plt.rcParams['figure.figsize'] = (2.9,2.8)
fig, ax = plt.subplots()

# Example data

people= ['Albedo','Crop_coeff','Crop_cover','CumGDD','EVI','Irr_nonirr','LW','prcp15','prcp30',
'prcp7','Soil','SolarZenith','SW','Tmax','VP','wind']

y_pos = np.arange(len(y))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, z, align='center',color='black')
#ax.barh(y_pos, z, align='edge')

ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importance')
ax.set_title('Predictors Importance (Non-Irrigated) RF_16',size=8)
#ax.set_title('Predictors Importance (Irrigated) RF_16',size=8)

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_xlim(0,0.4)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()




############################################################################################


############################################################################################

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
#df=pd.read_csv('predi_imp_irr.csv')
df=pd.read_csv('forcast_imp_irr.csv')

import os
import numpy as np
import matplotlib.pyplot as plt
x=list(df["imp"])
y=list(df["index"])

import matplotlib.pyplot as plt
#y = ['Physics','Chemistry','Economics','Philosophy','Computer Science','Sociology','Biology','History','Overall']
#x = [0.0,33.333333333333336,50.0,50.0,50.0,54.54545454545455,58.333333333333336,100.0,51.851851851851855]
#alternativeYlabels = ['0.02', '0.023', '0.024', '0.028', '0.029', '0.03', '0.031', '0.033', '0.037','0.038','0.043','0.044','0.08','0.11','0.13','0.29']plt.rcdefaults()
fig = plt.figure()
plt.rcParams['figure.figsize'] = (2.9,4.5)
fig, ax = plt.subplots()

# Example data

people= ['VP','SolarZenith','Crop_coeff','CumGDD','Irr_nonirr', 'SW_Day3','Crop_cover','EVI','prcp30',
'prcp_15','prcp_7','SW_Day2','Soil','Tmin_Day3','Tmax_Day3', 'prcp_Day3','SW_Day_present','prcp_Day2','Tmax_Day2',
'Tmax_Day1','LW_Day1','Tmin_Day1','Tmin_Day2','SW_Day1','wind','ALBEDO','prcp_Day1','tmax_F_present']

#people= ['-EVI,-Crop_coeff','-VP,-prcp30', '-SolarZenith,-CumGDD', '-SW_Day2,-SW_Day3']
        

y_pos = np.arange(len(y))
#performance = 3 + 10 * np.random.rand(len(people))
#error = np.random.rand(len(people))

ax.barh(y_pos, x, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importance')
ax.set_title('RF Forecast Model Predictors Importance',size=8)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_xlim(0,0.4)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

########################################################################################

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
#df=pd.read_csv('predi_imp_irr.csv')
df=pd.read_csv('forcast_imp_irr.csv')

import os
import numpy as np
import matplotlib.pyplot as plt
x=list(df["imp"])
y=list(df["index"])

y=['All predictors', '-EVI,-Crop_coeff','-VP,-prcp30', '-SolarZenith,-CumGDD', '-SW_Day2,-SW_Day3']
        
x=[0.56, 0.51,0.50,0.49,0.49]
import matplotlib.pyplot as plt
#y = ['Physics','Chemistry','Economics','Philosophy','Computer Science','Sociology','Biology','History','Overall']
#x = [0.0,33.333333333333336,50.0,50.0,50.0,54.54545454545455,58.333333333333336,100.0,51.851851851851855]
#alternativeYlabels = ['0.02', '0.023', '0.024', '0.028', '0.029', '0.03', '0.031', '0.033', '0.037','0.038','0.043','0.044','0.08','0.11','0.13','0.29']plt.rcdefaults()
fig = plt.figure()
plt.rcParams['figure.figsize'] = (2.9,3.5)
fig, ax = plt.subplots()

# Example data

#people= ['VP','SolarZenith','Crop_coeff','CumGDD','Irr_nonirr', 'SW_Day3','Crop_cover','EVI','prcp30',
#'prcp_15','prcp_7','SW_Day2','Soil','Tmin_Day3','Tmax_Day3', 'prcp_Day3','SW_Day_present','prcp_Day2','Tmax_Day2',
#'Tmax_Day1','LW_Day1','Tmin_Day1','Tmin_Day2','SW_Day1','wind','ALBEDO','prcp_Day1','tmax_F_present']

people= ['All predictors', '-EVI,-Crop_coeff','-VP,-prcp30', '-SolarZenith,-CumGDD', '-SW_Day2,-SW_Day3']
   

y_pos = np.arange(len(y))
#performance = 3 + 10 * np.random.rand(len(people))
#error = np.random.rand(len(people))

ax.barh(y_pos, x, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('NSE')
ax.set_title('LSTM Forecast Model Predictors Importance',size=8)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_xlim(0,1)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()







########################################################################################################
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']


df["resid1"]=(df.rftest_pred16-df.test_obs16)
df["resid2"]=(df.rftest_pred11-df.test_obs16)
df["resid3"]=(df.rftest_pred5-df.test_obs16)
df["resid4"]=(df.NOAA-df.test_obs16)

df.isnull().values.any()
apr=df[(df['month_n']=='Apr')]
may =df[(df['month_n']=='May')]
jun = df[(df['month_n']=='Jun')]
jul = df[(df['month_n']=='Jul')]
aug = df[(df['month_n']=='Aug')]
sep = df[(df['month_n']=='Sep')]
Oct = df[(df['month_n']=='Oct')]

my_dict1 = {'Apr':apr.resid1, 'May': may.resid1,'Jun': jun.resid1,'Jul': jul.resid1,'Aug': aug.resid1,'Sep': sep.resid1,'Oct': Oct.resid1}
my_dict2 = {'Apr':apr.resid2, 'May': may.resid2,'Jun': jun.resid2,'Jul': jul.resid2,'Aug': aug.resid2,'Sep': sep.resid2,'Oct': Oct.resid2}
my_dict3 = {'Apr':apr.resid3, 'May': may.resid3,'Jun': jun.resid3,'Jul': jul.resid3,'Aug': aug.resid3,'Sep': sep.resid3,'Oct': Oct.resid3}
my_dict4 = {'Apr':apr.resid4, 'May': may.resid4,'Jun': jun.resid4,'Jul': jul.resid4,'Aug': aug.resid4,'Sep': sep.resid4,'Oct': Oct.resid4}

c1="C0"
c2="C2"
c3="C1"
c4="C3"


plt.rcParams['figure.figsize'] = (3.5,2.5)
fig, ax = plt.subplots()

bp1=ax.boxplot(my_dict1.values(),sym='.',positions =[0,5,10,15,20,25,30],notch=True, widths=1, 
                 patch_artist=True,whiskerprops = dict(linestyle='--'
                           , linewidth=0.5), boxprops=dict(facecolor='blue'))
plt.setp(bp1["fliers"], markeredgecolor='blue')

bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [1,6,11,16,21,26,31],notch=True, widths=1, 
                 patch_artist=True, boxprops=dict(facecolor="pink"),whiskerprops = dict(linestyle='--'
                           , linewidth=0.5))

bp3=ax.boxplot(my_dict3.values(),sym='.',positions = [2,7,12,17,22,27,32],notch=True, widths=1, 
                 patch_artist=True, boxprops=dict(facecolor="yellow"),whiskerprops = dict(linestyle='--'
                           , linewidth=0.5))

bp4=ax.boxplot(my_dict4.values(),sym='.',positions = [3,8,13,18,23,28,33],notch=True, widths=1, 
                 patch_artist=True, boxprops=dict(facecolor="purple"),whiskerprops = dict(linestyle='--'
                           , linewidth=0.5))

plt.setp(bp1["boxes"], facecolor='blue',edgecolor='blue')
plt.setp(bp2["boxes"], facecolor='pink',edgecolor='pink')
plt.setp(bp3["boxes"], facecolor='yellow',edgecolor='yellow')
plt.setp(bp4["boxes"], facecolor='purple',edgecolor='purple')

plt.setp(bp1["fliers"], markeredgecolor='blue')
plt.setp(bp2["fliers"], markeredgecolor='pink')
plt.setp(bp3["fliers"], markeredgecolor='yellow')
plt.setp(bp4["fliers"], markeredgecolor='purple')

plt.xticks([2, 7, 11,16,22,27,32], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.legend([bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0],bp4["boxes"][0]], ['RF_16', 'RF_11','RF_5','NLDAS-Noah'], loc='lower right',prop={'size': 5})

ax.set_ylabel('Residuals (mm)')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_ylim(-8,4)

ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
#ax.locator_params(axis='y', nbins=6)
plt.axhline(y=0,linewidth=1, color='r', linestyle='--')
#plt.yticks(np.arange(-150, 150+1, 75))
#plt.show()
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
#####################################################################################################

#### LSTM residuls
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']

#### pick one residuals versus R2

## residuals
df["resid1"]=(df.rnntest_pred16-df.test_obs16)
df["resid2"]=(df.rnntest_pred11-df.test_obs16)
df["resid3"]=(df.rnntest_pred5-df.test_obs16)
df["resid4"]=(df.NOAA-df.test_obs16)
df.isnull().values.any()

apr=df[(df['month_n']=='Apr')]
may =df[(df['month_n']=='May')]

jun = df[(df['month_n']=='Jun')]
jul = df[(df['month_n']=='Jul')]
aug = df[(df['month_n']=='Aug')]
sep = df[(df['month_n']=='Sep')]
Oct = df[(df['month_n']=='Oct')]

my_dict1 = {'Apr':apr.resid1, 'May': may.resid1,'Jun': jun.resid1,'Jul': jul.resid1,'Aug': aug.resid1,'Sep': sep.resid1,'Oct': Oct.resid1}
my_dict2 = {'Apr':apr.resid2, 'May': may.resid2,'Jun': jun.resid2,'Jul': jul.resid2,'Aug': aug.resid2,'Sep': sep.resid2,'Oct': Oct.resid2}
my_dict3 = {'Apr':apr.resid3, 'May': may.resid3,'Jun': jun.resid3,'Jul': jul.resid3,'Aug': aug.resid3,'Sep': sep.resid3,'Oct': Oct.resid3}
my_dict4 = {'Apr':apr.resid4, 'May': may.resid4,'Jun': jun.resid4,'Jul': jul.resid4,'Aug': aug.resid4,'Sep': sep.resid4,'Oct': Oct.resid4}

c1="C0"
c2="C2"
c3="C1"
c4="C3"

plt.rcParams['figure.figsize'] = (3.5,2.5)

fig, ax = plt.subplots()
bp1=ax.boxplot(my_dict1.values(),sym='.',positions =[0,5,10,15,20,25,30],notch=True, widths=1, 
                 patch_artist=True,whiskerprops = dict(linestyle='--'
                           , linewidth=0.5), boxprops=dict(facecolor="C0"))
plt.setp(bp1["fliers"], markeredgecolor=c1)

bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [1,6,11,16,21,26,31],notch=True, widths=1, 
                 patch_artist=True, boxprops=dict(facecolor="C2"),whiskerprops = dict(linestyle='--'
                           , linewidth=0.5))

bp3=ax.boxplot(my_dict3.values(),sym='.',positions = [2,7,12,17,22,27,32],notch=True, widths=1, 
                 patch_artist=True, boxprops=dict(facecolor="C3"),whiskerprops = dict(linestyle='--'
                           , linewidth=0.5))

bp4=ax.boxplot(my_dict4.values(),sym='.',positions = [3,8,13,18,23,28,33],notch=True, widths=1, 
                 patch_artist=True, boxprops=dict(facecolor="C4"),whiskerprops = dict(linestyle='--'
                           , linewidth=0.5))

plt.setp(bp1["boxes"], facecolor='black',edgecolor='black')
plt.setp(bp2["boxes"], facecolor='orange',edgecolor='orange')
plt.setp(bp3["boxes"], facecolor='brown',edgecolor='brown')
plt.setp(bp4["boxes"], facecolor='purple',edgecolor='purple')

plt.setp(bp1["fliers"], markeredgecolor='black')
plt.setp(bp2["fliers"], markeredgecolor='orange')
plt.setp(bp3["fliers"], markeredgecolor='brown')
plt.setp(bp4["fliers"], markeredgecolor='purple')
plt.xticks([2, 7, 11,16,22,27,32], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0],bp4["boxes"][0]], ['rf_16', 'rf_11','rf_5','NOAA_LSM'], loc='upper left',prop={'size': 8})

ax.set_ylabel('Residuals (mm)')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.legend([bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0],bp4["boxes"][0]], ['LSTM_16', 'LSTM_11','LSTM_5','NLDAS-Noah'], loc='lower right',prop={'size': 5})

ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
plt.axhline(y=0,linewidth=1, color='r', linestyle='--')
ax.set_ylim(-8,4)
#plt.show()

ax = plt.gca()

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

##########################################################################################################

#### p  value diff between rf 11 and rf16
scipy.stats.f_oneway(apr.rftest_pred11,apr.rftest_pred16)

scipy.stats.f_oneway(may.rftest_pred11,may.rftest_pred16)

scipy.stats.f_oneway(jun.rftest_pred11,jun.rftest_pred16)

scipy.stats.f_oneway(jul.rftest_pred11,jul.rftest_pred16)

scipy.stats.f_oneway(aug.rftest_pred11,aug.rftest_pred16)

scipy.stats.f_oneway(sep.rftest_pred11,sep.rftest_pred16)

scipy.stats.f_oneway(Oct.rftest_pred11,Oct.rftest_pred16)

###overall diffference

scipy.stats.f_oneway(df.rftest_pred11,df.rftest_pred16)

#########################################################################################################

#### p  value diff between rf 16 and rnn16


scipy.stats.f_oneway(apr.rftest_pred16,apr.rnntest_pred16)

scipy.stats.f_oneway(may.rftest_pred16,may.rnntest_pred16)

scipy.stats.f_oneway(jun.rftest_pred16,jun.rnntest_pred16)

scipy.stats.f_oneway(jul.rftest_pred16,jul.rnntest_pred16)

scipy.stats.f_oneway(aug.rftest_pred16,aug.rnntest_pred16)

scipy.stats.f_oneway(sep.rftest_pred16,sep.rnntest_pred16)

scipy.stats.f_oneway(Oct.rftest_pred16,Oct.rnntest_pred16)


###overall
scipy.stats.f_oneway(df.rftest_pred16,df.rnntest_pred16)

#####################################################################################################
####  rf 11, rnn 11

scipy.stats.f_oneway(apr.rftest_pred11,apr.rnntest_pred11)

scipy.stats.f_oneway(may.rftest_pred11,may.rnntest_pred11)

scipy.stats.f_oneway(jun.rftest_pred11,jun.rnntest_pred11)

scipy.stats.f_oneway(jul.rftest_pred11,jul.rnntest_pred11)

scipy.stats.f_oneway(aug.rftest_pred11,aug.rnntest_pred11)

scipy.stats.f_oneway(sep.rftest_pred11,sep.rnntest_pred11)

scipy.stats.f_oneway(Oct.rftest_pred11,Oct.rnntest_pred11)

scipy.stats.f_oneway(df.rftest_pred11,df.rnntest_pred11)


#####################################################################################################
####  rnn 16, rnn 11

scipy.stats.f_oneway(apr.rnntest_pred16,apr.rnntest_pred11)

scipy.stats.f_oneway(may.rnntest_pred16,may.rnntest_pred11)

scipy.stats.f_oneway(jun.rnntest_pred16,jun.rnntest_pred11)

scipy.stats.f_oneway(jul.rnntest_pred16,jul.rnntest_pred11)

scipy.stats.f_oneway(aug.rnntest_pred16,aug.rnntest_pred11)

scipy.stats.f_oneway(sep.rnntest_pred16,sep.rnntest_pred11)

scipy.stats.f_oneway(Oct.rnntest_pred16,Oct.rnntest_pred11)

### overall

scipy.stats.f_oneway(df.rnntest_pred16,df.rnntest_pred11)
#########################################################################################################
#fig 3 residuals

#overall performance

### zoom in residuals 

os.chdir(r"C:/ammara_MD/ML_ET/results_paper1/PhD_dissertation/tables")
df=pd.read_csv('residuals.csv')

#yvals = df["Pine_ET_inches"]
#rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["rf16"]  # blue
lvals = df["rnn_16"]  ## black
mvals = df["rf11"]  ## green
nvals = df["rnn11"]  # oraange
pvals = df["rf_5"]  ## red
qvals = df["rnn_5"]## brown
rvals = df["NLDAS"]  ## purple

plt.rcParams['figure.figsize'] = (5, 2)
fig, ax = plt.subplots()
ax.set_ylabel('Residuals (mm)', color='black')
N = len(df["month"])
ind = np.arange(N)  # the x locations for the groups
width = 0.1       # the width of the bars

#patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
#rects1 = ax.bar(ind+width*1, zvals, width, color='blue')
#rects2 = ax.bar(ind+width*2, lvals, width, color='black')
#rects3 = ax.bar(ind+width*3, mvals, width, color='green')
#rects4 = ax.bar(ind+width*4, nvals, width, color='orange')
#rects5 = ax.bar(ind+width*5, pvals, width, color='red')
#rects6 = ax.bar(ind+width*6, qvals, width, color='brown')
#rects7 = ax.bar(ind+width*7, rvals, width, color='purple')

rects1 = ax.bar(ind+width*1, zvals, width, color='blue')
rects2 = ax.bar(ind+width*2, lvals, width, color='black')
rects3 = ax.bar(ind+width*3, mvals, width, color='pink')
rects4 = ax.bar(ind+width*4, nvals, width, color='orange')
rects5 = ax.bar(ind+width*5, pvals, width, color='yellow')
rects6 = ax.bar(ind+width*6, qvals, width, color='brown')
rects7 = ax.bar(ind+width*7, rvals, width, color='purple')

ax.set_xticks(ind+0.6)

#ax.set_xticks([1, 4, 6,8,10,12,14], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.set_xticklabels( ('Apr','May','Jun','Jul','Aug','Sep','Oct'))
#ax.set_ylim(0,8)
#ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0],rects7[0]), ('RF_16', 'LSTM_16','RF_11','LSTM_11','RF_5','LSTM_5','NLDAS_LSM') ,loc='lower left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0],rects7[0]), ('RF_16', 'LSTM_16','RF_11','LSTM_11','RF_5','LSTM_5','NLDAS-Noah') ,loc='lower left', bbox_to_anchor=(1, 0.1),prop={'size': 6})

#ax.legend()
#lgd=ax.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0],rects7[0]), ('rf_16', 'rf_11','LSTM_16','LSTM_11'),loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})


# pottaoes and wisp
#ax.legend( (rects1[0], rects2[0], rects3[0]), ('Pines','Wisp Model') ,loc='upper center', fontsize = 'large')
plt.axhline(y=0,linewidth=1, color='r', linestyle='--')
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
#plt.yticks(np.arange(-0.6,0.8))
#ax.set_ylim(-1,1)
ax.set_ylim(-1,0.5)

#plt.yticks(np.arange(-0.8,0.4+0.1,5))

#ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 

ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

########################################################################################################

### image for legend 

from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')


os.chdir(r"C:/ammara_MD/ML_ET/results_paper1/PhD_dissertation/tables")
df=pd.read_csv('residuals.csv')

plt.rcParams['figure.figsize'] = (4, 1.1)
fig, ax = plt.subplots()
ax.set_ylabel('Mean Residuals (mm)', color='black')
N = len(df["month"])
ind = np.arange(N)  # the x locations for the groups
width = 0.13       # the width of the bars
#yvals = df["Pine_ET_inches"]
#rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["rf16"]
kvals = df["ref11"]

lvals = df["rnn_16"]
mvals = df["rnn11"]
patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
rects2 = ax.bar(ind+width, zvals, width, color='blue')
rects3 = ax.bar(ind+width*2, kvals, width, color='green')
rects4 = ax.bar(ind+width*3, lvals, width, color='blue',hatch="**",edgecolor='black')
rects5 = ax.bar(ind+width*4, mvals, width, color='green',hatch="oo",edgecolor='black')
ax.set_xticks(ind+0.25)
ax.set_xticklabels( ('Apr','May','Jun','Jul','Aug','Sep','Oct'))
#ax.set_ylim(0,8)
ax.legend( (rects2[0], rects3[0],rects4[0],rects5[0]), ('rf_16', 'rf_11','LSTM_16','LSTM_11') )

#legend=ax.legend( (rects2[0], rects3[0],rects4[0],rects5[0]), ('rf_16', 'rf_11','LSTM_16','LSTM_11'))

legend=ax.legend( (rects2[0], rects3[0],rects4[0],rects5[0]), ('rf_16', 'rf_11','LSTM_16','LSTM_11'),loc='upper center', bbox_to_anchor=(1.4, 1),
          ncol=2, fancybox=True, shadow=True,prop={'size': 8})

# pottaoes and wisp
#ax.legend( (rects1[0], rects2[0], rects3[0]), ('Pines','Wisp Model') ,loc='upper center', fontsize = 'lar')
plt.axhline(y=0,linewidth=1, color='r', linestyle='--')
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
plt.yticks(np.arange(-20, 20+1,5))

#ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 

ax = plt.gca()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
plt.show()



def export_legend(legend, filename="legend.png", expand=[-4,-4,4,4]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=bbox)

export_legend(legend)
plt.show()

#########################################################################################################
def crop(df):
    if (df['crop'] =='corn'):
        return 'corn'
    elif (df['crop'] =='soy'):
        return 'soy'
    elif (df['crop'] =='potat1'):
        return 'potat'
    elif (df['crop'] =='potat2'):
        return 'potat'


os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
df['crop']=df.apply(crop,axis=1)

#df['marker']=df.apply(marker,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
a=df[(df['soil']=='sandy loam') & (df['crop']=='corn')]
b=df[(df['soil']=='loamy sand') & (df['crop']=='potat')]
c=df[(df['soil']=='loam') & (df['crop']=='corn')]
d=df[(df['soil']=='loam') & (df['crop']=='soy')]
e=df[(df['soil']=='silt loam') & (df['crop']=='corn')]
f=df[(df['soil']=='silt loam') & (df['crop']=='soy')]

# width height
plt.rcParams['figure.figsize'] = (3.2,3.1)
fig, ax = plt.subplots()
ax.plot(a['rftest_pred16'],a['test_obs16'],'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(b['rftest_pred16'],b['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Potatoes (Loamy Sand)")
ax.plot(c['rftest_pred16'],c['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['rftest_pred16'],d['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['rftest_pred16'],e['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['rftest_pred16'],f['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
#ax.plot(b['avg_ET3']*25.4,b['ET3_obs']*25.4,'>', color="black",markersize=,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend()
#lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
ax.set_title("RF_16 ET prediction model",size=10)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_ylim(0,12)
ax.set_xlim(0,12)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
a=df
pre=a['rnntest_pred5']
obs=a['test_obs16']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 5347 non irrigated
r2_score(obs,pre) ##  0.70


sqrt(mean_absolute_error(obs,pre))  #0.80
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias   #-4.7
###############################################################################################

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
df['crop']=df.apply(crop,axis=1)

#df['marker']=df.apply(marker,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
a=df[(df['soil']=='sandy loam') & (df['crop']=='corn')]
b=df[(df['soil']=='loamy sand') & (df['crop']=='potat')]
c=df[(df['soil']=='loam') & (df['crop']=='corn')]
d=df[(df['soil']=='loam') & (df['crop']=='soy')]
e=df[(df['soil']=='silt loam') & (df['crop']=='corn')]
f=df[(df['soil']=='silt loam') & (df['crop']=='soy')]

# width height
plt.rcParams['figure.figsize'] = (3.2,3.1)
fig, ax = plt.subplots()
ax.plot(a['NOAA'],a['test_obs16'],'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(b['NOAA'],b['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Potatoes (Loamy Sand)")
ax.plot(c['NOAA'],c['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['NOAA'],d['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['NOAA'],e['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['NOAA'],f['test_obs16'],'o', color="purple",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
#ax.plot(b['avg_ET3']*25.4,b['ET3_obs']*25.4,'>', color="black",markersize=,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend()
#lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
ax.set_title("NLDAS-Noah ET prediction model",size=10)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
ax.set_ylim(0,12)
ax.set_xlim(0,12)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

a=df
pre=a['NOAA']
obs=a['test_obs16']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 5347 non irrigated
r2_score(obs,pre) ##  0.57
sqrt(mean_absolute_error(obs,pre))  #0.90
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias   #0.3

##########################################################################################################
##time series plot for individual stations



############################################################################################################3
## figure C
import matplotlib.gridspec as gridspec
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df_MI=(df[(df['state']=='MI')])
df_NE=(df[(df['state']=='NE')])
df_MN=(df[(df['state']=='MN')])
df_IA=(df[(df['state']=='IA')])
df_WI=(df[(df['state']=='WI')])

df=df_MI
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
x1=(df["test_obs16"])
y1= (df["rftest_pred16"])
x2=(df["test_obs16"])
y2= (df["rnntest_pred16"])

df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])


Locator = mpl.dates.DayLocator(interval=60)
Formatter = mpl.dates.DateFormatter('%d-%m-%y')
fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)
fig.subplots_adjust(wspace=0.05)
fig.set_size_inches(3.5,1.3, forward=True)
ax.plot(df["TIMESTAMP"],x1, color="black",linestyle=':',label="Observed",linewidth=0.5)
ax.plot(df["TIMESTAMP"],y1, color="blue",linestyle=':',label="rf_16")
ax.plot(df["TIMESTAMP"],y2, color="green",linestyle=':',label="LSTM_16")
ax2.plot(df["TIMESTAMP"],x1, color="black",linestyle=':',label="Observed",linewidth=0.5)
ax2.plot(df["TIMESTAMP"],y1, color="blue",linestyle=':',label="rf_16")
ax2.plot(df["TIMESTAMP"],y2, color="green",linestyle=':',label="LSTM_16")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax2.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.set_ylim(0,12)
#MI 200, NE 300, MN 300, IW 250,WI 250





ax.set_xlim(datetime.datetime(2017, 4, 1), datetime.datetime(2017, 10, 31))
ax2.set_xlim(datetime.datetime(2018, 4, 1), datetime.datetime(2018, 10, 31))
labels = ax.get_xticklabels()
for label in labels:
    label.set_rotation(30)
labels = ax2.get_xticklabels()
for label in labels:
    label.set_rotation(30)
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.tick_params(right='off')
ax2.tick_params(left='off')
ax2.yaxis.tick_right()
ax.yaxis.tick_left()
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 4,direction='in')
ax2.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax2.tick_params(axis = 'x', which = 'major', labelsize = 4,direction='in')
ax.set_ylim(0,300)  # MI=200, NE,250
#ax.set_title("Location:Michigan,ID:US-KM1",size=5)
#ax2.set_title("Crop:Corn, Soil:Sandy Loam",size=5)

ax.locator_params(axis='y', nbins=4)
ax.locator_params(axis='x', nbins=4)
ax2.locator_params(axis='y', nbins=4)
ax2.locator_params(axis='x', nbins=4)
ax.xaxis.set_major_locator(copy.copy(Locator))
ax.xaxis.set_major_formatter(copy.copy(Formatter))
ax2.xaxis.set_major_locator(copy.copy(Locator))
ax2.xaxis.set_major_formatter(copy.copy(Formatter))
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#########################################################################################################
def crop(df):
    if (df['crop'] =='corn'):
        return 'corn'
    elif (df['crop'] =='soy'):
        return 'soy'
    elif (df['crop'] =='potat1'):
        return 'potat'
    elif (df['crop'] =='potat2'):
        return 'potat'

##### 
import matplotlib.gridspec as gridspec
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df_MI_T3=(df[(df['ID']=='MI_T3')])
df_NB_Ne2=(df[(df['ID']=='NB_Ne2')])
df_MN_Ro3=(df[(df['ID']=='MN_Ro3')])
df_IA_Br1=(df[(df['ID']=='IA_Br1')])
df_WI_CS=(df[(df['ID']=='WI_CS')])
df_MN_Ro1=(df[(df['ID']=='MN_Ro1')])



fin=df_MN_Ro1
obs=fin.test_obs16
pre=fin.rnntest_pred16
#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
r2_score(obs,pre)

sqrt(mean_squared_error(obs,pre))

pbias=(np.sum(((obs-pre)**2))/np.sum(obs))*100

pbias
####################################################################################################

##### training


########################################################################################################
#### train
import matplotlib.gridspec as gridspec
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('fore_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import brier_score_loss

def r2_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmse = np.sqrt( mean_squared_error( g['ET3_obs']*25.4, g['rnnL3ET3']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return pd.Series( dict(rmse = rmse ) )

df.groupby( 'ID' ).apply( r2_rmse ).reset_index()
##########################################################################################################

### long way to do this
df_NB_Ne1=(df[(df['ID']=='NB_Ne1')])
df_MI_nonirri=(df[(df['ID']=='MI_nonirri')])
df_IL_Bo1=(df[(df['ID']=='IL_Bo1')])
df_MN_Ro5=(df[(df['ID']=='MN_Ro5')])
df_IL_B02=(df[(df['ID']=='IL_B02')])
df_MI_irrig=(df[(df['ID']=='MI_irrig')])
df_OH_CRT=(df[(df['ID']=='OH_CRT')])
df_MN_Ro2=(df[(df['ID']=='MN_Ro2')])
df_MI_T4=(df[(df['ID']=='MI_T4')])
df_NB_Ne3=(df[(df['ID']=='NB_Ne3')])
df_IA_Br3=(df[(df['ID']=='IA_Br3')])
df_MN_Ro6=(df[(df['ID']=='MN_Ro6')])
df_IL_1B1=(df[(df['ID']=='IL_1B1')])
df_MI_T3=(df[(df['ID']=='MI_T3')])

fin=df_IL_1B1
obs=fin.ET2_obs
pre=fin.rfL3ET2
sqrt(mean_squared_error(obs,pre))
###################################################################################################
# long way to do this
import matplotlib.gridspec as gridspec
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('fore_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']

def r2_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmse = np.sqrt( mean_squared_error( g['ET3_obs'], g['rnnL3ET3'] ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return pd.Series( dict(rmse = rmse ) )

df.groupby( 'ID' ).apply( r2_rmse ).reset_index()

#######################################################################################################
### long way to do this
df_MI_T3=(df[(df['ID']=='MI_T3')])
df_NB_Ne2=(df[(df['ID']=='NB_Ne2')])
df_MN_Ro3=(df[(df['ID']=='MN_Ro3')])
df_IA_Br1=(df[(df['ID']=='IA_Br1')])
df_WI_CS=(df[(df['ID']=='WI_CS')])
df_MN_Ro1=(df[(df['ID']=='MN_Ro1')])


fin=df_MN_Ro1
obs=fin.ET2_obs
pre=fin.rfL3ET2
#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
#r2_score(obs,pre)
sqrt(mean_squared_error(obs,pre))
###################################################################################################


Index(['ID', 'TIMESTAMP', 'DOY', 'month', 'week', 'crop', 'irr_nonirr',
       'ET1_obs', 'ET2_obs', 'ET3_obs', 'rfL1ET1', 'rfL1ET2', 'rfL1ET3',
       'rfL2ET1', 'rfL2ET2', 'rfL2ET3', 'rfL3ET1', 'rfL3ET2', 'rfL3ET3',
       'rnnL1ET1', 'rnnL1ET2', 'rnnL1ET3', 'rnnL2ET1', 'rnnL2ET2', 'rnnL2ET3',
       'rnnL3ET1', 'rnnL3ET2', 'rnnL3ET3'],
      dtype='object')

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('fore_test.csv')
######################################################################################################3
orig=df
list_months=[5]
df=df[pd.to_datetime(df['TIMESTAMP']).dt.month.isin(list_months)]
may=df

df=orig
list_months=[6]
df=df[pd.to_datetime(df['TIMESTAMP']).dt.month.isin(list_months)]
jun=df

df=orig
list_months=[7]
df=df[pd.to_datetime(df['TIMESTAMP']).dt.month.isin(list_months)]
jul=df

df=orig
list_months=[8]
df=df[pd.to_datetime(df['TIMESTAMP']).dt.month.isin(list_months)]
aug=df
#######################################################################################################
df=corn
df.soil.unique()
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
x1=(df["test_obs16"][(df['soil']=='sandy loam')])*25.4
y1= (df["rnntest_pred16"][(df['soil']=='sandy loam')])*25.4
x2=(df["test_obs16"][(df['soil']=='silt loam')])*25.4
y2= (df["rnntest_pred16"][(df['soil']=='silt loam')])*25.4
x3=(df["test_obs16"][(df['soil']=='loam')])*25.4
y3=(df["rnntest_pred16"][(df['soil']=='loam')])*25.4

# width height
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(x1,y1,'x', color="indigo",markersize=2,label="Sandy Loam")
ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 6})
#ax.set_title("Corn",size=10)
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.yaxis.label.set_size(5) 
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,300)
ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()


###############################################################################################3333

import matplotlib.gridspec as gridspec
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('fore_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
corn = df[(df['crop']=='corn')]
soy = df[(df['crop']=='soy')]
potat=df.loc[df['crop'].isin(['potat1','potat2'])]

####################################################################################################
#### get hot versus wet yaers

## plot for triangular taylor


import matplotlib.lines as mlines
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5,4)
fig, ax = plt.subplots()


black_plus = mlines.Line2D([], [], color='black', marker='+', linestyle='none',
                          markersize=10, label='Corn Day 1 ET',markeredgewidth=2.0)

black_o = mlines.Line2D([], [], color='black', marker='o', mfc='none', linestyle='None',
                          markersize=10, label='Corn Day 2 ET',markeredgewidth=2.0)

black_cro = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=10, label='Corn Day 3 ET',markeredgewidth=2.0)

red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='none',
                          markersize=10, label='Soybeans Day 1 ET',markeredgewidth=2.0)

red_o = mlines.Line2D([], [], color='red', marker='o', mfc='none', linestyle='None',
                          markersize=10, label='Soybeans Day 2 ET',markeredgewidth=2.0)

red_cro = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                          markersize=10, label='Soybeans Day 3 ET',markeredgewidth=2.0)

gr_plus = mlines.Line2D([], [], color='green', marker='+', linestyle='none',
                          markersize=10, label='Potatoes Day 1 ET',markeredgewidth=2.0)

gr_o = mlines.Line2D([], [], color='green', marker='o', mfc='none', linestyle='None',
                          markersize=10, label='Potatoes Day 2 ET',markeredgewidth=2.0)

gr_cro = mlines.Line2D([], [], color='green', marker='x', linestyle='None',
                          markersize=10, label='Potatoes Day 3 ET',markeredgewidth=2.0)


legend=plt.legend(handles=[black_plus,black_o, black_cro,red_plus,red_o, red_cro,gr_plus,gr_o, gr_cro])

plt.show()



def export_legend(legend, filename="legend.png", expand=[-1,-1,1,1]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=bbox)

export_legend(legend)
plt.show()


#####################################################################################################33


### box plot for forecasting data  not much difference between day 1 day 2 day 3 ET

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('fore_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']

df["resL1ET1"]=(df.rfL1ET1-df.ET1_obs)*25.4
df["resL1ET2"]=(df.rfL1ET2-df.ET2_obs)*25.4
df["resL1ET3"]=(df.rfL1ET3-df.ET3_obs)*25.4

#df["resid1"]=(df.rnntest_pred16-df.test_obs16)*25.4
#df["resid2"]=(df.rnntest_pred11-df.test_obs16)*25.4

df.isnull().values.any()

apr=df[(df['month_n']=='Apr')]
may =df[(df['month_n']=='May')]
jun = df[(df['month_n']=='Jun')]
jul = df[(df['month_n']=='Jul')]
aug = df[(df['month_n']=='Aug')]
sep = df[(df['month_n']=='Sep')]
Oct = df[(df['month_n']=='Oct')]

my_dict1 = {'Apr':apr.resL1ET1, 'May': may.resL1ET1,'Jun': may.resL1ET1,'Jul': may.resL1ET1,'Aug': may.resL1ET1,'Sep': may.resL1ET1,'Oct': may.resL1ET1}
my_dict2 = {'Apr':apr.resL1ET2, 'May': may.resL1ET2,'Jun': may.resL1ET2,'Jul': may.resL1ET2,'Aug': may.resL1ET2,'Sep': may.resL1ET2,'Oct': may.resL1ET2}
my_dict3 = {'Apr':apr.resL1ET3, 'May': may.resL1ET3,'Jun': may.resL1ET3,'Jul': may.resL1ET3,'Aug': may.resL1ET3,'Sep': may.resL1ET3,'Oct': may.resL1ET3}


c1="C0"
c2="C2"
c3="C3"
plt.rcParams['figure.figsize'] = (3.5,3.5)


fig, ax = plt.subplots()
bp1=ax.boxplot(my_dict1.values(),sym='.',positions =[0,2,4,6,8,10,12],notch=True, widths=0.7, 
                 patch_artist=True, boxprops=dict(facecolor="C0"))
plt.setp(bp1["fliers"], markeredgecolor=c1)
#plt.setp(box1["whiskers"], markeredgecolor=c1)
#bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [0.5,2.5,4.5,6.5,8.5,10.5,12.5],notch=True, widths=1, 
 #                patch_artist=True, boxprops=dict(facecolor="C2"))
bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [0.7,2.7,4.7,6.7,8.7,10.7,12.7],notch=True, widths=0.7, 
                 patch_artist=True, boxprops=dict(facecolor="C2"))

plt.setp(bp2["fliers"], markeredgecolor=c2)


bp3=ax.boxplot(my_dict3.values(),sym='.',positions = [1,3,5,7,9,11,13],notch=True, widths=0.7, 
                 patch_artist=True, boxprops=dict(facecolor="C3"))

plt.setp(bp2["fliers"], markeredgecolor=c3)



#ax.set_xticklabels(my_dict1.keys())
plt.xticks([0, 2, 4,6,8,10,12], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['rf_16', 'rf_11'], loc='upper left',prop={'size': 12})

ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],['ET Day 1', 'ET Day 2', 'ET Day 3'], loc='upper left',prop={'size': 8})
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['LSTM_16', 'LSTM_11'], loc='upper left',prop={'size':8})

ax.set_ylabel('Residuals (mm)')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')

ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
#ax.locator_params(axis='y', nbins=6)
plt.axhline(y=0,linewidth=2, color='r', linestyle='-')
#ax.set_ylim(-150,150)  non truncated
ax.set_ylim(-150,150)  # truncated


#plt.show()
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################

### by crop, by ID

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('fore_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
corn = df[(df['crop']=='corn')]
soy = df[(df['crop']=='soy')]
potat=df.loc[df['crop'].isin(['potat1','potat2'])]


L1_avgET_pr=corn[['rfL1ET1','rfL1ET2','rfL1ET3']].mean(axis=1)
L2_avgET_pr=corn[['rfL2ET1','rfL2ET2','rfL2ET3']].mean(axis=1)
L3_avgET_pr=corn[['rfL3ET1','rfL3ET2','rfL3ET3']].mean(axis=1)


avgET_obs=corn[['ET1_obs','ET2_obs','ET3_obs']].mean(axis=1)
corn_res_L1=(L1_avgET_pr-avgET_obs)*25.4
corn_res_L2=(L2_avgET_pr-avgET_obs)*25.4
corn_res_L3=(L3_avgET_pr-avgET_obs)*25.4


L1_avgET_pr=soy[['rfL1ET1','rfL1ET2','rfL1ET3']].mean(axis=1)
L2_avgET_pr=soy[['rfL2ET1','rfL2ET2','rfL2ET3']].mean(axis=1)
L3_avgET_pr=soy[['rfL3ET1','rfL3ET2','rfL3ET3']].mean(axis=1)

avgET_obs=soy[['ET1_obs','ET2_obs','ET3_obs']].mean(axis=1)
soy_res_L1=(L1_avgET_pr-avgET_obs)*25.5
soy_res_L2=(L2_avgET_pr-avgET_obs)*25.4
soy_res_L3=(L3_avgET_pr-avgET_obs)*25.4


avgET_obs=potat[['ET1_obs','ET2_obs','ET3_obs']].mean(axis=1)
L1_avgET_pr=potat[['rfL1ET1','rfL1ET2','rfL1ET3']].mean(axis=1)
L2_avgET_pr=potat[['rfL2ET1','rfL2ET2','rfL2ET3']].mean(axis=1)
L3_avgET_pr=potat[['rfL3ET1','rfL3ET2','rfL3ET3']].mean(axis=1)

pot_res_L1=(L1_avgET_pr-avgET_obs)*25.5
pot_res_L2=(L2_avgET_pr-avgET_obs)*25.4
pot_res_L3=(L3_avgET_pr-avgET_obs)*25.4



cornlevels={'cornL1':corn_res_L1,'cornL2':corn_res_L2,'cornL3':corn_res_L3}
soylevels ={'soyL1':soy_res_L1,'soyL2':soy_res_L2,'soyL3':soy_res_L3}
potlevels ={'potL1':pot_res_L1,'potL2':pot_res_L2,'potL3':pot_res_L3}

#,'soy':soy_res_L1,'pot':potat_res_L1
#corn_dict1 = {'L1ET1':soy_avg_L1}
#corn_dict1 = {'cornL1ET1':corn.resL1ET1, 'L2ET1 ': corn.resL2ET1,'L3ET1':corn.resL3ET1,'L1ET2':corn.resL1ET2,'L2ET2 ': corn.resL2ET2,'L3ET2':corn.resL3ET2,'L1ET3':corn.resL1ET3,'L2ET3 ': corn.resL2ET3,'L3ET3':corn.resL3ET3}     
    
#my_dict2 = {'Apr':apr.resL1ET2, 'May': may.resL1ET2,'Jun': may.resL1ET2,'Jul': may.resL1ET2,'Aug': may.resL1ET2,'Sep': may.resL1ET2,'Oct': may.resL1ET2}
#my_dict3 = {'Apr':apr.resL1ET3, 'May': may.resL1ET3,'Jun': may.resL1ET3,'Jul': may.resL1ET3,'Aug': may.resL1ET3,'Sep': may.resL1ET3,'Oct': may.resL1ET3}

#level_dict1 = {'L1':corn_avg_L1}

c1="C0"
c2="C2"
c3="C3"
plt.rcParams['figure.figsize'] = (4.5,3.5)


fig, ax = plt.subplots()
#bp1=ax.boxplot(corn_dict1.values(),sym='.',positions =[0,2,4,6,8,10,12,14,16],notch=True, widths=0.7, 
 #                patch_artist=True, boxprops=dict(facecolor="C0"))

bp1=ax.boxplot(cornlevels.values(),sym='.',positions =[0,1,2],notch=True, widths=0.7, 
                 patch_artist=True, boxprops=dict(facecolor="C0"))

plt.setp(bp1["fliers"], markeredgecolor=c1)

bp2=ax.boxplot(soylevels.values(),sym='.',positions =[4,5,6],notch=True, widths=0.7, 
                 patch_artist=True, boxprops=dict(facecolor="C2"))

plt.setp(bp2["fliers"], markeredgecolor=c2)

bp3=ax.boxplot(potlevels.values(),sym='.',positions =[8,9,10],notch=True, widths=0.7, 
                 patch_artist=True, boxprops=dict(facecolor="C3"))

plt.setp(bp3["fliers"], markeredgecolor=c3)

plt.xticks([0, 1, 2,4,5,6,8,9,10], ['L1','L2','L3','L1','L2','L3','L1','L2','L3'])

ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],['Corn (n=4257)', 'Soy (n=3258)', 'Potatoes (n=218)'], loc='upper left',prop={'size': 8})
ax.set_ylabel('Residuals (mm)')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')

ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
plt.axhline(y=0,linewidth=2, color='r', linestyle='-')
ax.set_ylim(-200,200)  # truncated
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
###########################################################################################################

##find extremes 

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('fore_test.csv')
dat=pd.read_csv('forcdates.test.csv')
df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
df['prcp']=dat['prcp']
df['tmax_F']=dat['tmax_F']
df['year'] = pd.DatetimeIndex(df['TIMESTAMP']).year
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
orig=df
#########################################################################################################

#### use these lines to look for extremes condition
test = df.groupby([df.year, df.crop,df.ID]).mean()
test=test.sort_values(by=['tmax_F'])
###############################################################################################
### if start over
df=orig
df = df[(df['ID']=='NB_Ne2')]
df= df[(df['year']==2012)]  # 2412, 2423
#NB_dry.index = np.arange(len(NB_dry))
stra=df.iloc[92:95]  
stra.index = np.arange(1,len(stra)+1)     
plt.plot(stra.TIMESTAMP,stra.ET1_obs,'r')
end=df.iloc[95:98]  
#df.index = np.arange(1,len(df)+1)     
plt.plot(df.rfL1ET1,'b')
        
#########################################################################################################

###jacknife plot     

 ## 0.73 confidence for day 1 ET       

 ## 0.73 confidence for day 2 ET       
        
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('jacknife.csv')
import numpy as np
import matplotlib.pyplot as plt

df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['month_n']=df.apply(month_n,axis=1)
#df['marker']=df.apply(marker,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
corn = df[(df['crop']=='corn')]
soy = df[(df['crop']=='soy')]
potat=df.loc[df['crop'].isin(['potat1','potat2'])]
     
#fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
#ax.locator_params(nbins=4)
#ax = axs[0,1]
#ax.errorbar(df.D1y_test, df.D1y_hat, yerr=np.sqrt(df.D1unbiased), fmt='o')
#ax.set_title('Hor. symmetric')
df=corn
df=(df[(df['soil']=='sandy loam')])

df=(df[(df['soil']=='silt loam')])
df=(df[(df['soil']=='loam')])


df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.apply(pd.to_numeric, errors='coerce')
df=df.astype(float) 
df=df.rename_axis('TIME').reset_index() 
df=df.resample('M', on='TIME').mean()
#df=df.set_index('TIMESTAMP')  
#m1= df.resample('M')
df = df[pd.notnull(df['D2unbiased'])]

#df.soil.unique()
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
df_san_lo=df
df_sil_lo=df
df_loa=df



#############################################################################################################
### error for day 3

plt.rcParams['figure.figsize'] = (4.2,2.1)
fig, ax = plt.subplots()

#a3=df_sil_lo.sort_values(['v_3']).reset_index()
#plt.scatter(a3.index,a3.v_3*25.4,color='red',marker='o',label="Observed",s=8)


a4=df_sil_lo.sort_values(['v_3']).reset_index()
plt.scatter(a4.index,a4.v_3*25.4,color='red',marker='o',s=8)
plt.errorbar(a4.index,a4.p_3*25.4,yerr=[a4.p_3*25.4-a4.p_d_3*25.4,a4.p_u_3*25.4-a4.p_3*25.4], c='blue',
         Label="Silt Loam",capthick=2,fmt='o',markersize='3')


a3=df_san_lo.sort_values(['v_3']).reset_index()
plt.scatter(a3.index,a3.v_3*25.4,color='red',marker='o',s=8)
plt.errorbar(a3.index,a3.p_3*25.4,yerr=[a3.p_3*25.4-a3.p_d_3*25.4,a3.p_u_3*25.4-a3.p_3*25.4], c='green', 
          label="Sandy Loam",capthick=2,fmt='o',markersize='3')

a3=df_loa.sort_values(['v_3']).reset_index()
plt.scatter(a3.index,a3.v_3*25.4,color='red',marker='o',s=8)
plt.errorbar(a3.index,a3.p_3*25.4,yerr=[a3.p_3*25.4-a3.p_d_3*25.4,a3.p_u_3*25.4-a3.p_3*25.4], c='black',
          label="Loam",capthick=2,fmt='o',markersize='3')

line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 6})
#ax.set_title("Potatoes",size=10)
ax.set_xlabel('Sample Size')  # we already handled the x-label with ax1
ax.set_ylabel('Monthly Average Predicted ET (mm)')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,200)
ax.set_xlim(0,70)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

#####################################################################################################


import matplotlib.lines as mlines
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5,4)
fig, ax = plt.subplots()


black_plus = mlines.Line2D([], [], color='red', marker='o', linestyle='none',
                          markersize=6, label='Observed',markeredgewidth=2.0)

black_o = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=6, label='Silt Laom',markeredgewidth=2.0)

black_cro = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=6, label='Sandy Laom',markeredgewidth=2.0)

red_plus = mlines.Line2D([], [], color='black', marker='o', linestyle='none',
                          markersize=6, label='Loam',markeredgewidth=2.0)

legend=plt.legend(handles=[black_plus,black_o, black_cro,red_plus])

plt.show()



def export_legend(legend, filename="legend.png", expand=[-1,-1,1,1]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=bbox)

export_legend(legend)
plt.show()



























































###########################################################################################################




































#array(['silt loam', 'loam'], dtype=object) 

x4=df["test_obs16"][(df['soil']=='loamy sand')]*25.4
y4= df["rftest_pred16"][(df['soil']=='loamy sand')]*25.4

ax.plot(x4,y4,'p', color="violet",markersize=1,label='Loamy Sand')
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)

line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 6})
#ax.set_title("Potatoes",size=10)
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,150)
ax.set_xlim(0,150)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()







##############################################################################################################

### back up
plt.scatter(a3.index,a3.v_3*25.4,color='red',marker='o')
plt.errorbar(a3.index,a3.p_3*25.4,yerr=[a3.p_3*25.4-a3.p_d_3*25.4,a3.p_u_3*25.4-a3.p_3*25.4], fmt='+',c='g')

a3=df_sil_lo.sort_values(['v_3']).reset_index()
plt.scatter(a3.index,a3.v_3*25.4,color='red',marker='o')
plt.errorbar(a3.index,a3.p_3*25.4,yerr=[a3.p_3*25.4-a3.p_d_3*25.4,a3.p_u_3*25.4-a3.p_3*25.4], fmt='+',c='b')


a3=df_loa.sort_values(['v_3']).reset_index()
plt.scatter(a3.index,a3.v_3*25.4,color='red',marker='o')
plt.errorbar(a3.index,a3.p_3*25.4,yerr=[a3.p_3*25.4-a3.p_d_3*25.4,a3.p_u_3*25.4-a3.p_3*25.4], fmt='+',c='black')









#############################################################################################################

plt.errorbar(df.D1y_test, df.D1y_hat, yerr=np.sqrt(df.D1unbiased), fmt='o',c='r')
plt.errorbar(df.D2y_test, df.D2y_hat, yerr=np.sqrt(df.D2unbiased), fmt='+',c='b')
plt.errorbar(df.D3y_test, df.D3y_hat, yerr=np.sqrt(df.D3unbiased), fmt='p',c='g')

#############################################################################################################
### error for day 1
a1=df.sort_values(['v_1']).reset_index()
plt.scatter(a1.index,a1.v_1,color='red',marker='o')
plt.errorbar(a1.index,a1.p_1,yerr=[a1.p_1-a1.p_d_1,a1.p_u_1-a1.p_1], fmt='o',c='g')
###########################################################################################################
### error for day 2
a2=df.sort_values(['v_2']).reset_index()
plt.scatter(a2.index,a2.v_2,color='red',marker='o')
plt.errorbar(a2.index,a2.p_3,yerr=[a2.p_2-a2.p_d_2,a2.p_u_2-a2.p_2], fmt='o',c='g')


















plt.xlabel('Reported MPG')
plt.ylabel('Predicted MPG')
plt.show()



import nitime.algorithms as tsa



import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats.distributions as dist

import nitime.algorithms as tsa
import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate




























apr=df[(df['month_n']=='Apr')]
may =df[(df['month_n']=='May')]
jun = df[(df['month_n']=='Jun')]
jul = df[(df['month_n']=='Jul')]
aug = df[(df['month_n']=='Aug')]
sep = df[(df['month_n']=='Sep')]
Oct = df[(df['month_n']=='Oct')]

plt.errorbar(apr.D1y_test, apr.D1y_hat, yerr=np.sqrt(apr.D1unbiased), fmt='o')
#plt.plot([5, 45], [5, 45], 'k--')
plt.xlabel('Reported MPG')
plt.ylabel('Predicted MPG')
plt.show()








df.soil.unique()
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
x1=(df["test_obs16"][(df['soil']=='sandy loam')])*25.4
y1= (df["rnntest_pred16"][(df['soil']=='sandy loam')])*25.4
x2=(df["test_obs16"][(df['soil']=='silt loam')])*25.4
y2= (df["rnntest_pred16"][(df['soil']=='silt loam')])*25.4
x3=(df["test_obs16"][(df['soil']=='loam')])*25.4
y3=(df["rnntest_pred16"][(df['soil']=='loam')])*25.4

# width height
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(x1,y1,'x', color="indigo",markersize=2,label="Sandy Loam")
ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': 6})
#ax.set_title("Corn",size=10)
ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,300)
ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()



















