# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:57:02 2020

@author: Ammara
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:02:25 2020

@author: Ammara
"""

#https://seaborn.pydata.org/generated/seaborn.lineplot.html
import itertools 
#import chart_studio.plotly
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
from sklearn.metrics import mean_absolute_error

from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
#import seaborn as sns; sns.set()
import matplotlib.pyplot as plt



def NS(s,o):
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

def crop_df(df):
    if (df['crop'] =='corn'):
        return 'corn'
    elif (df['crop'] =='soy'):
        return 'soy'
    elif (df['crop'] =='potat1'):
        return 'potat'
    elif (df['crop'] =='potat2'):
        return 'potat'

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\forecast_results")

df = pd.read_excel('L1_ann.xlsx', sheetname='forc_test')


df_test_rnn_L1=df

df = pd.read_excel('L2_ann.xlsx', sheetname='forc_test')
df_test_rnn_L2=df

df = pd.read_excel('L3_ann.xlsx', sheetname='forc_test')
df_test_rnn_L3=df
##################################################################################################
### random forest train/test level 1
df = pd.read_excel('L1_rf.xlsx', sheetname='forc_for_train')
df_train_rf_L1=df
df = pd.read_excel('L1_rf.xlsx', sheetname='forc_test')
df_test_rf_L1=df
##################################################################################################
### random forest train/test level 2
df = pd.read_excel('L2_rf.xlsx', sheetname='forc_for_train')
df_train_rf_L2=df
df = pd.read_excel('L2_rf.xlsx', sheetname='forc_test')
df_test_rf_L2=df
##################################################################################################
### random forest train/test level 3
df = pd.read_excel('L3_rf.xlsx', sheetname='forc_for_train')
df_train_rf_L3=df
df = pd.read_excel('L3_rf.xlsx', sheetname='forc_test')
df_test_rf_L3=df
##################################################################################################
## level 1
corn_L1 = df_test_rf_L1[(df_test_rf_L1['crop']=='corn')]
soy_L1 = df_test_rf_L1[(df_test_rf_L1['crop']=='soy')]
potat_L1=df_test_rf_L1.loc[df_test_rf_L1['crop'].isin(['potat1','potat2'])]

##################################################################################################
## level 2
corn_L2 = df_test_rf_L2[(df_test_rf_L2['crop']=='corn')]
soy_L2 = df_test_rf_L2[(df_test_rf_L2['crop']=='soy')]
potat_L2=df_test_rf_L2.loc[df_test_rf_L2['crop'].isin(['potat1','potat2'])]

##################################################################################################
## level 3
corn_L3 = df_test_rf_L3[(df_test_rf_L3['crop']=='corn')]
soy_L3 = df_test_rf_L3[(df_test_rf_L3['crop']=='soy')]
potat_L3=df_test_rf_L3.loc[df_test_rf_L3['crop'].isin(['potat1','potat2'])]
################################################################################################
## let's just try on test data level 1 crop and soil type 
df=df_test_rf_L3
#df=df_test_rf_L2
#df=df_test_rf_L3
#array(['MI_T3', 'NB_Ne2', 'MN_Ro3', 'IA_Br1', 'WI_CS', 'MN_Ro1'],

df['soil'] = df.apply(soil_df,axis=1)
df["crop"]=df.apply(crop_df,axis=1)
#df=df[df['ID'] == 'MI_T3']
#df.set_index('TIMESTAMP',inplace=True)
#####################################33
#month=df.index.month
#df=df.groupby(month).mean()

###############################################3
####### perterbation spread
# 


#### start with forcasting plot prof. desai mentioned 

## ET day 1
df=df_test_rnn_L1

df["avg"]=(df["rfL1ET1cntr"]+df["rfL1ET1per1"]+df["rfL1ET1per2"]+df["rfL1ET1per3"]+df["rfL1ET1per4"]
+df["rfL1ET1per5"]+df["rfL1ET1per6"]+df["rfL1ET1per7"]+df["rfL1ET1per8"]+df["rfL1ET1per9"]
+df["rfL1ET1per10"])/11

#df["std"]=(df["rfL1ET1cntr"]-df["ET1_obs"]).std()


#############################################################################################
### resid  ensem-obs
############################################################################################

df["resid"]=((df["rfL1ET1cntr"]-df["ET1_obs"])+(df["rfL1ET1per1"]-df["ET1_obs"])+(df["rfL1ET1cntr"]-df["ET1_obs"])
+(df["rfL1ET1per1"]-df["ET1_obs"])+(df["rfL1ET1per2"]-df["ET1_obs"])+(df["rfL1ET1per3"]-df["ET1_obs"])
+(df["rfL1ET1per4"]-df["ET1_obs"])+(df["rfL1ET1per5"]-df["ET1_obs"])+(df["rfL1ET1per6"]-df["ET1_obs"])
+(df["rfL1ET1per7"]-df["ET1_obs"])+(df["rfL1ET1per8"]-df["ET1_obs"])+(df["rfL1ET1per9"]-df["ET1_obs"])
+(df["rfL1ET1per10"]-df["ET1_obs"]))/11


s=df.groupby( 'month').mean()
a=s["avg"]
b=s["avg"]-s["resid"]
c=s["avg"]+s["resid"]

#### try filling 
x=[4,5,6,7,8,9,10]
plt.plot(x, s["avg"], 'k-')
plt.fill_between(x, s["avg"]-s["resid"], s["avg"]+s["resid"])
#plt.fill_between(x,  s["avg"]+s["resid"])

plt.show()


#### try ET day 2 with level 2

df=df_test_rnn_L2

df["avg"]=(df["rfL2ET2cntr"]+df["rfL2ET2per1"]+df["rfL2ET2per2"]+df["rfL2ET2per3"]+df["rfL2ET2per4"]
+df["rfL2ET2per5"]+df["rfL2ET2per6"]+df["rfL2ET2per7"]+df["rfL2ET2per8"]+df["rfL2ET2per9"]
+df["rfL2ET2per10"])/11

L2_ET2_sim=df["avg"]
L2_ET2_obs=df["ET2_obs"]

#df["std"]=(df["rfL1ET1cntr"]-df["ET1_obs"]).std()

#############################################################################################
### resid  ensem-obs
############################################################################################

df["resid"]=((df["rfL2ET2cntr"]-df["ET2_obs"])+(df["rfL2ET2per1"]-df["ET2_obs"])+(df["rfL2ET2cntr"]-df["ET2_obs"])
+(df["rfL2ET2per1"]-df["ET2_obs"])+(df["rfL2ET2per2"]-df["ET2_obs"])+(df["rfL2ET2per3"]-df["ET2_obs"])
+(df["rfL2ET2per4"]-df["ET2_obs"])+(df["rfL2ET2per5"]-df["ET2_obs"])+(df["rfL2ET2per6"]-df["ET2_obs"])
+(df["rfL2ET2per7"]-df["ET2_obs"])+(df["rfL2ET2per8"]-df["ET2_obs"])+(df["rfL2ET2per9"]-df["ET2_obs"])
+(df["rfL2ET2per10"]-df["ET2_obs"]))/11

###############################################################################################
## anothor way to show error

df["std"]=((df["rfL2ET2cntr"]-df["ET2_obs"])+(df["rfL2ET2per1"]-df["ET2_obs"])+(df["rfL2ET2cntr"]-df["ET2_obs"])
+(df["rfL2ET2per1"]-df["ET2_obs"])+(df["rfL2ET2per2"]-df["ET2_obs"])+(df["rfL2ET2per3"]-df["ET2_obs"])
+(df["rfL2ET2per4"]-df["ET2_obs"])+(df["rfL2ET2per5"]-df["ET2_obs"])+(df["rfL2ET2per6"]-df["ET2_obs"])
+(df["rfL2ET2per7"]-df["ET2_obs"])+(df["rfL2ET2per8"]-df["ET2_obs"])+(df["rfL2ET2per9"]-df["ET2_obs"])
+(df["rfL2ET2per10"]-df["ET2_obs"]))


df_std=pd.concat((df["rfL2ET2cntr"],df["rfL2ET2per1"],df["rfL2ET2per2"],df["rfL2ET2per3"]
,df["rfL2ET2per4"],df["rfL2ET2per5"],df["rfL2ET2per6"],df["rfL2ET2per7"],df["rfL2ET2per8"],df["rfL2ET2per9"]
,df["rfL2ET2per10"]),axis=1)


df["std"]=df_std.std(axis=1)


############################################################################################3
### spread based on ensemble-avg sim
###########################################################################################

s=df.groupby( 'month').mean()
a=s["avg"]
b=s["avg"]-s["resid"]
c=s["avg"]+s["resid"]

#### try filling 
x=[4,5,6,7,8,9,10]
plt.plot(x, s["avg"], 'k-')
plt.plot(x, s["ET3_obs"])


plt.plot(x, s["avg"], 'k-')
plt.plot(x, s["ET3_obs"], 'r-')

#plt.fill_between(x, s["avg"]-s["resid"], s["avg"]+s["resid"])
plt.fill_between(x, s["avg"]-s["std"], s["avg"]+s["std"])

plt.show()

###############################################################################################

#### try ET day 3 Level 3 rf

df=df_test_rnn_L2
df=df_test_rf_L2



### day1
df["avg"]=(df["rfL1ET1cntr"]+df["rfL1ET1per1"]+df["rfL1ET1per2"]+df["rfL1ET1per3"]+df["rfL1ET1per4"]
+df["rfL1ET1per5"]+df["rfL1ET1per6"]+df["rfL1ET1per7"]+df["rfL1ET1per8"]+df["rfL1ET1per9"]
+df["rfL1ET1per10"])/11

obs=df["ET1_obs"]
pre=df["avg"]


### day2
df["avg"]=(df["rfL2ET2cntr"]+df["rfL2ET2per1"]+df["rfL2ET2per2"]+df["rfL2ET2per3"]+df["rfL2ET2per4"]
+df["rfL2ET2per5"]+df["rfL2ET2per6"]+df["rfL2ET2per7"]+df["rfL2ET2per8"]+df["rfL2ET2per9"]
+df["rfL2ET2per10"])/11

## day 3
df["avg"]=(df["rfL3ET3cntr"]+df["rfL3ET3per1"]+df["rfL3ET3per2"]+df["rfL3ET3per3"]+df["rfL3ET3per4"]
+df["rfL3ET3per5"]+df["rfL3ET3per6"]+df["rfL3ET3per7"]+df["rfL3ET3per8"]+df["rfL3ET3per9"]
+df["rfL3ET3per10"])/11

#df["std"]=(df["rfL1ET1cntr"]-df["ET1_obs"]).std()

#############################################################################################
### resid  ensem-obs
############################################################################################

#df["resid"]=((df["rfL3ET3cntr"]-df["ET3_obs"])+(df["rfL3ET3per1"]-df["ET3_obs"])+(df["rfL3ET3cntr"]-df["ET3_obs"])
#+(df["rfL3ET3per1"]-df["ET3_obs"])+(df["rfL3ET3per2"]-df["ET3_obs"])+(df["rfL3ET3per3"]-df["ET3_obs"])
#+(df["rfL3ET3per4"]-df["ET3_obs"])+(df["rfL3ET3per5"]-df["ET3_obs"])+(df["rfL3ET3per6"]-df["ET3_obs"])
#+(df["rfL3ET3per7"]-df["ET3_obs"])+(df["rfL3ET3per8"]-df["ET3_obs"])+(df["rfL3ET3per9"]-df["ET3_obs"])
#+(df["rfL3ET3per10"]-df["ET3_obs"]))/11

## anothor way to show error

## day 3
df_std=pd.concat((df["rfL3ET3cntr"],df["rfL3ET3per1"],df["rfL3ET3per2"],df["rfL3ET3per3"]
,df["rfL3ET3per4"],df["rfL3ET3per5"],df["rfL3ET3per6"],df["rfL3ET3per7"],df["rfL3ET3per8"],df["rfL3ET3per9"]
,df["rfL3ET3per10"]),axis=1)

df["std"]=df_std.std(axis=1)


### day 1
df_std=pd.concat((df["rfL1ET1cntr"],df["rfL1ET1per1"],df["rfL1ET1per2"],df["rfL1ET1per3"]
,df["rfL1ET1per4"],df["rfL1ET1per5"],df["rfL1ET1per6"],df["rfL1ET1per7"],df["rfL1ET1per8"],df["rfL1ET1per9"]
,df["rfL1ET1per10"]),axis=1)

df["std"]=df_std.std(axis=1)


### day 2
df_std=pd.concat((df["rfL2ET2cntr"],df["rfL2ET2per1"],df["rfL2ET2per2"],df["rfL2ET2per3"]
,df["rfL2ET2per4"],df["rfL2ET2per5"],df["rfL2ET2per6"],df["rfL2ET2per7"],df["rfL2ET2per8"],df["rfL2ET2per9"]
,df["rfL2ET2per10"]),axis=1)

df["std"]=df_std.std(axis=1)


s=df
s=df.groupby( 'month').mean()
x=[4,5,6,7,8,9,10]

############################################################################################
### spread based on ensemble-avg sim
####################################################
#a=s["avg"]
#b=s["avg"]-s["resid"]
#c=s["avg"]+s["resid"]

#### try filling 
x=[4,5,6,7,8,9,10]

plt.rcParams['figure.figsize'] = (4.2,3.1)
fig, ax = plt.subplots()
ax.plot(x, s["ET1_obs"], 'r-',label="Observed ET")
#ax.plot(x, s["avg"], 'k-',label="Day 3 Model Forecast Average")
#ax.plot(x, s["avg"], 'k-',label="Day 2 Model Forecast Average")
ax.plot(x, s["avg"], 'k-',label="Day 1 Model Forecast Average")

ax.fill_between(x,s["avg"]-s["std"], s["avg"]+s["std"],alpha=0.4,label="Ensembles Standard Deviation")
#ax.fill_between(x,s["avg"]-s["avg"].std(), s["avg"]+s["avg"].std(),alpha=0.4)
ax.plot(x, s["avg"], 'o', color='tab:brown')
#ax.set_title("Corn",size=10)
#ax.set_ylabel('Day 3 ET Forecast (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('Day 1 ET Forecast (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('Day 2 ET Forecast (mm)')  # we already handled the x-label with ax1

#ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
#ax.legend(loc='lower center',prop={'size': 8})
ax.set_title("RF Forecast Model",size=10)
#ax.set_title("LSTM Forecast Model",size=10)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.legend()
lgd=ax.legend(loc='lower center',fontsize="8")
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_ylim(0,5)
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([4, 5, 6,7,8,9,10], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

############################################################################################
obs=df["ET1_obs"]
pre=df["avg"]

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)

r2_score(obs,pre)
#(mean_squared_error(obs,pre))
mean_absolute_error(obs,pre) 
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

##########################################################################################3

#### try ET day 2 with level 1

df=df_test_rf_L1

df["avg"]=(df["rfL1ET2cntr"]+df["rfL1ET2per1"]+df["rfL1ET2per2"]+df["rfL1ET2per3"]+df["rfL1ET2per4"]
+df["rfL1ET2per5"]+df["rfL1ET2per6"]+df["rfL1ET2per7"]+df["rfL1ET2per8"]+df["rfL1ET2per9"]
+df["rfL1ET2per10"])/11

L1_ET2_sim=df["avg"]
L1_ET2_obs=df["ET2_obs"]

#df["std"]=(df["rfL1ET1cntr"]-df["ET1_obs"]).std()

#############################################################################################
### resid  ensem-obs
############################################################################################

df["resid"]=((df["rfL1ET2cntr"]-df["ET2_obs"])+(df["rfL1ET2per1"]-df["ET2_obs"])+(df["rfL1ET2cntr"]-df["ET2_obs"])
+(df["rfL1ET2per1"]-df["ET2_obs"])+(df["rfL1ET2per2"]-df["ET2_obs"])+(df["rfL1ET2per3"]-df["ET2_obs"])
+(df["rfL1ET2per4"]-df["ET2_obs"])+(df["rfL1ET2per5"]-df["ET2_obs"])+(df["rfL1ET2per6"]-df["ET2_obs"])
+(df["rfL1ET2per7"]-df["ET2_obs"])+(df["rfL1ET2per8"]-df["ET2_obs"])+(df["rfL1ET2per9"]-df["ET2_obs"])
+(df["rfL1ET2per10"]-df["ET2_obs"]))/11

###################################################
### spread based on ensemble-avg sim
####################################################

s=df.groupby( 'month').mean()
a=s["avg"]
b=s["avg"]-s["resid"]
c=s["avg"]+s["resid"]

#### try filling 
x=[4,5,6,7,8,9,10]
plt.plot(x, s["avg"], 'k-')
plt.fill_between(x, s["avg"]-s["resid"], s["avg"]+s["resid"])
plt.show()

#############################################################################################


###############################################################################################

#### try ET day 3 Level 1

df=df_test_rf_L1

df["avg"]=(df["rfL1ET3cntr"]+df["rfL1ET3per1"]+df["rfL1ET3per2"]+df["rfL1ET3per3"]+df["rfL1ET3per4"]
+df["rfL1ET3per5"]+df["rfL1ET3per6"]+df["rfL1ET3per7"]+df["rfL1ET3per8"]+df["rfL1ET3per9"]
+df["rfL1ET3per10"])/11

#df["std"]=(df["rfL1ET1cntr"]-df["ET1_obs"]).std()

#############################################################################################
### resid  ensem-obs
############################################################################################

df["resid"]=((df["rfL1ET3cntr"]-df["ET3_obs"])+(df["rfL1ET3per1"]-df["ET3_obs"])+(df["rfL1ET3cntr"]-df["ET3_obs"])
+(df["rfL1ET3per1"]-df["ET3_obs"])+(df["rfL1ET3per2"]-df["ET3_obs"])+(df["rfL1ET3per3"]-df["ET3_obs"])
+(df["rfL1ET3per4"]-df["ET3_obs"])+(df["rfL1ET3per5"]-df["ET3_obs"])+(df["rfL1ET3per6"]-df["ET3_obs"])
+(df["rfL1ET3per7"]-df["ET3_obs"])+(df["rfL1ET3per8"]-df["ET3_obs"])+(df["rfL1ET3per9"]-df["ET3_obs"])
+(df["rfL1ET3per10"]-df["ET3_obs"]))/11

###################################################
### spread based on ensemble-avg sim
####################################################
s=df.groupby( 'month').mean()
a=s["avg"]
b=s["avg"]-s["resid"]
c=s["avg"]+s["resid"]

#### try filling 
x=[4,5,6,7,8,9,10]
plt.plot(x, s["avg"], 'k-')
plt.fill_between(x, s["avg"]-s["resid"], s["avg"]+s["resid"])
plt.show()

###############################################################################################























#############################################################################################
## level 2 ET

df=df_test_rf_L2

##############################################################################################
def ET2_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2cntr']*25.4 ) )
    rmseper1 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per1']*25.4 ) )
    rmseper2 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per2']*25.4 ) )
    rmseper3 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per3']*25.4 ) )
    rmseper4 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per4']*25.4 ) )
    rmseper5 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per5']*25.4 ) )
    rmseper6 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per6']*25.4 ) )
    rmseper7 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per7']*25.4 ) )
    rmseper8 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per8']*25.4 ) )
    rmseper9 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per9']*25.4 ) )
    rmseper10 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL2ET2per10']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return ([rmsecntr,rmseper1,rmseper2,rmseper3,rmseper4,rmseper5,rmseper6,rmseper7,rmseper8,rmseper9,rmseper10])

s=df.groupby( 'month' ).apply(ET2_rmse)

df=df_test_rf_L3

def ET3_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3cntr']*25.4 ) )
    rmseper1 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per1']*25.4 ) )
    rmseper2 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per2']*25.4 ) )
    rmseper3 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per3']*25.4 ) )
    rmseper4 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per4']*25.4 ) )
    rmseper5 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per5']*25.4 ) )
    rmseper6 =( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per6']*25.4 ) )
    rmseper7 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per7']*25.4 ) )
    rmseper8 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per8']*25.4 ) )
    rmseper9 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per9']*25.4 ) )
    rmseper10 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL2ET3per10']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return ([rmsecntr,rmseper1,rmseper2,rmseper3,rmseper4,rmseper5,rmseper6,rmseper7,rmseper8,rmseper9,rmseper10])

s=df.groupby( 'month' ).apply(ET3_rmse)

############################################################################################

## level 3 ET
df=df_test_rf_L3


def ET3_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3cntr']*25.4 ) )
    rmseper1 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per1']*25.4 ) )
    rmseper2 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per2']*25.4 ) )
    rmseper3 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per3']*25.4 ) )
    rmseper4 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per4']*25.4 ) )
    rmseper5 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per5']*25.4 ) )
    rmseper6 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per6']*25.4 ) )
    rmseper7 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per7']*25.4 ) )
    rmseper8 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per8']*25.4 ) )
    rmseper9 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per9']*25.4 ) )
    rmseper10 = (mean_absolute_error(g['ET3_obs']*25.4, g['rfL3ET3per10']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return ([rmsecntr,rmseper1,rmseper2,rmseper3,rmseper4,rmseper5,rmseper6,rmseper7,rmseper8,rmseper9,rmseper10])

s=df.groupby( 'month' ).apply(ET3_rmse)







###################################################################################################


s=os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\PhD_dissertation")

df = pd.read_excel('results_rf_MAE.xlsx', sheetname='level_ET1')
df_res_L1=df

df = pd.read_excel('results_rf_MAE.xlsx', sheetname='level_ET2')
df_res_L2=df

df = pd.read_excel('results_rf_MAE.xlsx', sheetname='level_ET3')
df_res_L3=df

s=os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\PhD_dissertation")

df = pd.read_excel('results_rnn_MAE.xlsx', sheetname='level_ET1')
df_resrn_L1=df

df = pd.read_excel('results_rnn_MAE.xlsx', sheetname='level_ET2')
df_resrn_L2=df

df = pd.read_excel('results_rnn_MAE.xlsx', sheetname='level_ET3')
df_resrn_L3=df


################################################################################################
# width length
plt.rcParams['figure.figsize'] = (2.0, 1.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('MAE (mm)', color='black')
ax1.set_xlabel('Scenarios', color='black')
df=df_resrn_L1
plt.plot(df["ID"],df["aug_ET1"],linestyle="-",label="Day 1 Forecast (F.Met day 1)",color='green',markersize=2,marker='o')
#plt.plot(df["ID"],df["jun_ET2"],linestyle="-",label="Day 2 Forecast (F.Met day 1)",color='blue',markersize=2,marker='p')
df=df_resrn_L2
plt.plot(df["ID"],df["aug_L2_ET2"],linestyle="-",label="Day 2 Forecast (F.Met day 1,2)",color='orange',markersize=2,marker='<')
df=df_resrn_L1
#plt.plot(df["ID"],df["jun_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1)",color='red',markersize=2,marker='s')
df=df_resrn_L3
#plt.plot(df["ID"],df["jun_L2_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1,2)",color='purple',markersize=2,marker='8')
plt.plot(df["ID"],df["aug_L3_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1,2,3)",color='black',markersize=2,marker='v')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
lgd=ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 4.5})
ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
#ax1.set_ylim(27.0,32.0)
#ax1.set_ylim(32.0,42.0)
#ax1.set_ylim(26.0,34.0)
#ax1.set_ylim(22,26)
#ax1.set_ylim(26,32)
ax1.set_ylim(20,26)

ax1.set_title("LSTM ET Forecasting (August)",fontsize=6)
ax1.set_xticklabels( ('Cntr','per1','per2','per3','per4','per5','per6','per7','per8','per9','per10'))
ax1.set_xticks(np.arange(1,12))
fig.autofmt_xdate() 

plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
#################################################################################################
################################################################################################
# width length
plt.rcParams['figure.figsize'] = (2.0, 1.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('MAE (mm)', color='black')
ax1.set_xlabel('Scenarios', color='black')
df=df_res_L1
plt.plot(df["ID"],df["aug_ET1"],linestyle="-",label="Day 1 Forecast (F.Met day 1)",color='green',markersize=2,marker='o')
#plt.plot(df["ID"],df["jun_ET2"],linestyle="-",label="Day 2 Forecast (F.Met day 1)",color='blue',markersize=2,marker='p')
df=df_res_L2
plt.plot(df["ID"],df["aug_L2_ET2"],linestyle="-",label="Day 2 Forecast (F.Met day 1,2)",color='orange',markersize=2,marker='<')
df=df_res_L1
#plt.plot(df["ID"],df["jun_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1)",color='red',markersize=2,marker='s')
df=df_res_L3
#plt.plot(df["ID"],df["jun_L2_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1,2)",color='purple',markersize=2,marker='8')
plt.plot(df["ID"],df["aug_L3_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1,2,3)",color='black',markersize=2,marker='v')
ax1.tick_params(axis='y', labelcolor='black')
ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
#ax1.set_ylim(26.0,34.0)
#ax1.set_ylim(27.0,32.0)
#ax1.set_ylim(32.0,42.0)
#ax1.set_ylim(22,26)
ax1.set_ylim(20,26)

ax1.set_title("rf ET Forecasting (August)",fontsize=6)
ax1.set_xticklabels( ('Cntr','per1','per2','per3','per4','per5','per6','per7','per8','per9','per10'))
ax1.set_xticks(np.arange(1,12))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
#################################################################################################
### effect of Foreast meterology

plt.rcParams['figure.figsize'] = (2.0, 1.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('MAE (mm)', color='black')
ax1.set_xlabel('Scenarios', color='black')
#df=df_resrn_L1
#plt.plot(df["ID"],df["aug_ET1"],linestyle="-",label="Day 1 Forecast (F.Met day 1)",color='green',markersize=2,marker='o')
#plt.plot(df["ID"],df["jun_ET2"],linestyle="-",label="Day 2 Forecast (F.Met day 1)",color='blue',markersize=2,marker='p')
#df=df_resrn_L2
#plt.plot(df["ID"],df["aug_L2_ET2"],linestyle="-",label="Day 2 Forecast (F.Met day 1,2)",color='orange',markersize=2,marker='<')
df=df_resrn_L1
plt.plot(df["ID"],df["jun_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1)",color='red',markersize=2,marker='s')
df=df_resrn_L3
plt.plot(df["ID"],df["jun_L2_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1,2)",color='purple',markersize=2,marker='8')
plt.plot(df["ID"],df["aug_L3_ET3"],linestyle="-",label="Day 3 Forecast (F.Met day 1,2,3)",color='black',markersize=2,marker='v')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
lgd=ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 4.5})
ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
#ax1.set_ylim(27.0,32.0)
#ax1.set_ylim(32.0,42.0)
#ax1.set_ylim(26.0,34.0)
#ax1.set_ylim(22,26)
#ax1.set_ylim(26,32)
ax1.set_ylim(20,26)

ax1.set_title("LSTM ET Forecasting (August)",fontsize=6)
ax1.set_xticklabels( ('Cntr','per1','per2','per3','per4','per5','per6','per7','per8','per9','per10'))
ax1.set_xticks(np.arange(1,12))
fig.autofmt_xdate() 

plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()














##################################################################################################
### random forest train/test level 1
s=os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\forecast_results")
df = pd.read_excel('L1_rf.xlsx', sheetname='forc_test')
df_test_rf_L1=df
##################################################################################################
### random forest train/test level 2
df = pd.read_excel('L2_rf.xlsx', sheetname='forc_test')
df_test_rf_L2=df
##################################################################################################
### random forest train/test level 3
df = pd.read_excel('L3_rf.xlsx', sheetname='forc_test')
df_test_rf_L3=df
####################################################################################################
#

def crop(df):
    if (df['crop'] =='corn'):
        return 'corn'
    elif (df['crop'] =='soy'):
        return 'soy'
    elif (df['crop'] =='potat1'):
        return 'potat'
    elif (df['crop'] =='potat2'):
        return 'potat'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np



df=df_test_rf_L1

df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)

df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET1"]=(df['rfL1ET1cntr'])
df["residET1"]=(df["avg_ET1"]-df["ET1_obs"])
df_ET1=df
df=df_test_rf_L2
df['soil'] = df.apply(soil_df,axis=1)
df['crop'] = df.apply(crop,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET2"]=(df['rfL2ET2cntr'])
df["residET2"]=(df["avg_ET2"]-df["ET2_obs"])
df_ET2=df

df=df_test_rf_L3
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET3"]=(df['rfL3ET3cntr'])
df["residET3"]=(df["avg_ET3"]-df["ET3_obs"])
df_ET3=df

df=pd.concat((df_ET1['ID'],df_ET1['TIMESTAMP'],df_ET1['DOY'],df_ET1['month'],df_ET1['crop']
,df_ET1['irr_nonirr'],df_ET1['ET1_obs'],df_ET1['ET2_obs'],df_ET1['ET3_obs'],df_ET1["avg_ET1"],df_ET2["avg_ET2"],df_ET3["avg_ET3"],df_ET1['soil'],
df_ET1['residET1'],df_ET2['residET2'],df_ET3['residET3']),axis=1)

rf_irr_nonirr=df


corn_sanL=df[(df['soil']=='sandy loam') & (df['crop']=='corn')]
pot_lomsa=df[(df['soil']=='loamy sand') & (df['crop']=='potat')]
corn_lom=df[(df['soil']=='loam') & (df['crop']=='corn')]
soy_lom=df[(df['soil']=='loam') & (df['crop']=='soy')]
corn_siltL=df[(df['soil']=='silt loam') & (df['crop']=='corn')]
soy_siltL=df[(df['soil']=='silt loam') & (df['crop']=='soy')]
##################################################################################################
plt.figure(1)
labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

plt.rcParams['figure.figsize'] = (3.0, 2.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Residuals (mm)', color='black')
#ax1.set_xlabel('Scenarios', color='black')

positions = [1]
data = [corn_sanL['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Sandy Loam")    

positions = [2]
data = [pot_lomsa['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Loamy Sand")    

positions = [3]
data = [corn_lom['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Loam")    

positions = [4]
data = [soy_lom['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Loam")    

positions = [5]
data = [soy_siltL['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Silt Loam")    

positions = [6]
data = [corn_siltL['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Silt Loam")    
ax1.tick_params(axis='y', labelcolor='black')
ax1.yaxis.label.set_size(8)
ax1.xaxis.label.set_size(8) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 8)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax1.set_ylim(-10,10)

ax1.set_title("RF ET day 3 Forecasting",fontsize=10)
ax1.set_xticklabels( ('Corn','Potatoes','Corn','Soybeans','Soybeans','Corn'))
ax1.set_xticks(np.arange(1,7))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig2.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

############################################################################################
##rnn violin
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\forecast_results")
df = pd.read_excel('L1_ann.xlsx', sheetname='forc_test')
df_test_rnn_L1=df
df = pd.read_excel('L2_ann.xlsx', sheetname='forc_test')
df_test_rnn_L2=df
df = pd.read_excel('L3_ann.xlsx', sheetname='forc_test')
df_test_rnn_L3=df

######################################################################################################
df=df_test_rnn_L1
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET1"]=(df['rfL1ET1cntr'])
df["residET1"]=(df["avg_ET1"]-df["ET1_obs"])
#df["avg_ET1"]=(df['rfL1ET1cntr']+df['rfL1ET1per1']+df['rfL1ET1per2']+df['rfL1ET1per3']+df['rfL1ET1per4']+df['rfL1ET1per5']
#+df['rfL1ET1per6']+df['rfL1ET1per7']+df['rfL1ET1per8']+df['rfL1ET1per9']+df['rfL1ET1per10'])/10
#df["residET1"]=(df["avg_ET1"]-df["ET1_obs"])*25.4
df_ET1=df

df=df_test_rnn_L2
df['soil'] = df.apply(soil_df,axis=1)
df['crop'] = df.apply(crop,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET2"]=(df['rfL2ET2cntr'])
df["residET2"]=(df["avg_ET2"]-df["ET2_obs"])
df_ET2=df

df=df_test_rnn_L3
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET3"]=(df['rfL3ET3cntr'])
df["residET3"]=(df["avg_ET3"]-df["ET3_obs"])
df_ET3=df

df=pd.concat((df_ET1['ID'],df_ET1['TIMESTAMP'],df_ET1['DOY'],df_ET1['month'],df_ET1['crop']
,df_ET1['irr_nonirr'],df_ET1['ET1_obs'],df_ET1['ET2_obs'],df_ET1['ET3_obs'],df_ET1["avg_ET1"],df_ET2["avg_ET2"],df_ET3["avg_ET3"],df_ET1['soil'],
df_ET1['residET1'],df_ET2['residET2'],df_ET3['residET3']),axis=1)
lstmirr_nonirr=df


corn_sanL=df[(df['soil']=='sandy loam') & (df['crop']=='corn')]
pot_lomsa=df[(df['soil']=='loamy sand') & (df['crop']=='potat')]
corn_lom=df[(df['soil']=='loam') & (df['crop']=='corn')]
soy_lom=df[(df['soil']=='loam') & (df['crop']=='soy')]
corn_siltL=df[(df['soil']=='silt loam') & (df['crop']=='corn')]
soy_siltL=df[(df['soil']=='silt loam') & (df['crop']=='soy')]
##################################################################################################

plt.figure(1)
labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

plt.rcParams['figure.figsize'] = (3.0, 2.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Residuals (mm)', color='black')
#ax1.set_xlabel('Scenarios', color='black')
positions = [1]
data = [corn_sanL['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Sandy Loam")    
positions = [2]
data = [pot_lomsa['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Loamy Sand")    
positions = [3]
data = [corn_lom['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Loam")    
positions = [4]
data = [soy_lom['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Loam")    
positions = [5]
data = [soy_siltL['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Silt Loam")    
positions = [6]
data = [corn_siltL['residET3'].values]
add_label(plt.violinplot(data, positions,showmeans=True), "Silt Loam")    
ax1.tick_params(axis='y', labelcolor='black')
ax1.yaxis.label.set_size(8)
ax1.xaxis.label.set_size(8) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 8)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax1.set_ylim(-10,10)
plt.legend(*zip(*labels),loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6.5})
ax1.set_title("LSTM ET day 3 Forecasting ",fontsize=10)
ax1.set_xticklabels( ('Corn','Potatoes','Corn','Soybeans','Soybeans','Corn'))
ax1.set_xticks(np.arange(1,7))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig2.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
##################################################################################################

###irrigated_non irrigated

#lstmirr_nonirr
#rfirr_nonirr
df=lstmirr_nonirr
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
a=df[(df['soil']=='sandy loam') & (df['crop']=='corn')&(df['irr_nonirr']==0)]
c=df[(df['soil']=='loam') & (df['crop']=='corn')&(df['irr_nonirr']==0)]
d=df[(df['soil']=='loam') & (df['crop']=='soy')&(df['irr_nonirr']==0)]
e=df[(df['soil']=='silt loam') & (df['crop']=='corn')&(df['irr_nonirr']==0)]
f=df[(df['soil']=='silt loam') & (df['crop']=='soy')&(df['irr_nonirr']==0)]

# width height
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(a['avg_ET3'],a['ET3_obs'],'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
#ax.plot(c['avg_ET3'],c['ET3_obs'],'p', color="seagreen",markersize=2,fillstyle='none',label="Corn (Loam)")
#ax.plot(d['avg_ET3'],d['ET3_obs'],'s', color="gold",markersize=2,fillstyle='none',label="Soybeans (Loam)")
#ax.plot(e['avg_ET3'],e['ET3_obs'],'h', color="red",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
#ax.plot(f['avg_ET3'],f['ET3_obs'],'v', color="brown",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")

ax.plot(a['avg_ET3'],a['ET3_obs'],'o', color="blue",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(c['avg_ET3'],c['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['avg_ET3'],d['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3'],e['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3'],f['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")

#ax.plot(b['avg_ET3']*25.4,b['ET3_obs']*25.4,'>', color="black",markersize=,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend()
#lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
ax.set_title("LSTM Day 3 ET Forecast (Non Irrigated)",size=7)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=8)
ax.locator_params(axis='x', nbins=8)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
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

a=df[(df['irr_nonirr']==0)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 5347 non irrigated
r2_score(obs,pre) ##  0.47
sqrt(mean_absolute_error(obs,pre))  #23.2
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias   #-1.53
###############################################################################################

df=lstmirr_nonirr
#array(['sandy loam', 'silt loam', 'loam'], dtype=object) #      'clay loam'], dtype=object) 
a=df[(df['soil']=='sandy loam') & (df['crop']=='corn')&(df['irr_nonirr']==1)]
b=df[(df['soil']=='loamy sand') & (df['crop']=='potat')&(df['irr_nonirr']==1)]
c=df[(df['soil']=='loam') & (df['crop']=='corn')&(df['irr_nonirr']==1)]
d=df[(df['soil']=='loam') & (df['crop']=='soy')&(df['irr_nonirr']==1)]
e=df[(df['soil']=='silt loam') & (df['crop']=='corn')&(df['irr_nonirr']==1)]
f=df[(df['soil']=='silt loam') & (df['crop']=='soy')&(df['irr_nonirr']==1)]

# width height
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(a['avg_ET3']*25.4,a['ET3_obs']*25.4,'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
#ax.plot(c['avg_ET3']*25.4,c['ET3_obs']*25.4,'p', color="seagreen",markersize=2,fillstyle='none',label="Corn (Loam)")
#ax.plot(d['avg_ET3']*25.4,d['ET3_obs']*25.4,'s', color="gold",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3'],e['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3'],f['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
ax.plot(b['avg_ET3'],b['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_title("LSTM Day 3 ET Forecast (Irrigated)",size=7)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=8)
ax.locator_params(axis='x', nbins=8)
#ax.legend()
#lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
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

a=df[(df['irr_nonirr']==1)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 2386  irrigated
r2_score(obs,pre) ##  0.67
sqrt(mean_absolute_error(obs,pre))  #21.9
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias   #-5.15

##################################################################################################
## random forest irrigated
################################################################################################
df=rf_irr_nonirr

a=df[(df['soil']=='sandy loam') & (df['crop']=='corn')&(df['irr_nonirr']==0)]
c=df[(df['soil']=='loam') & (df['crop']=='corn')&(df['irr_nonirr']==0)]
d=df[(df['soil']=='loam') & (df['crop']=='soy')&(df['irr_nonirr']==0)]
e=df[(df['soil']=='silt loam') & (df['crop']=='corn')&(df['irr_nonirr']==0)]
f=df[(df['soil']=='silt loam') & (df['crop']=='soy')&(df['irr_nonirr']==0)]
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(a['avg_ET3'],a['ET3_obs'],'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
#ax.plot(c['avg_ET3'],c['ET3_obs'],'p', color="seagreen",markersize=2,fillstyle='none',label="Corn (Loam)")
#ax.plot(d['avg_ET3'],d['ET3_obs'],'s', color="gold",markersize=2,fillstyle='none',label="Soybeans (Loam)")
#ax.plot(e['avg_ET3'],e['ET3_obs'],'h', color="red",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
#ax.plot(f['avg_ET3'],f['ET3_obs'],'v', color="brown",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")

ax.plot(a['avg_ET3'],a['ET3_obs'],'o', color="blue",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(c['avg_ET3'],c['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['avg_ET3'],d['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3'],e['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3'],f['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend()
#lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
ax.set_title("RF Day 3 ET Forecast (non Irrigated)",size=7)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=8)
ax.locator_params(axis='x', nbins=8)

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
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

a=df[(df['irr_nonirr']==0)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 2386 non irrigated
r2_score(obs,pre) ##  0.53
sqrt(mean_absolute_error(obs,pre))  #22.4
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias   #-5.1

###############################################################################################

df=rf_irr_nonirr

a=df[(df['soil']=='sandy loam') & (df['crop']=='corn')&(df['irr_nonirr']==1)]
b=df[(df['soil']=='loamy sand') & (df['crop']=='potat')&(df['irr_nonirr']==1)]
c=df[(df['soil']=='loam') & (df['crop']=='corn')&(df['irr_nonirr']==1)]
d=df[(df['soil']=='loam') & (df['crop']=='soy')&(df['irr_nonirr']==1)]
e=df[(df['soil']=='silt loam') & (df['crop']=='corn')&(df['irr_nonirr']==1)]
f=df[(df['soil']=='silt loam') & (df['crop']=='soy')&(df['irr_nonirr']==1)]
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
ax.plot(a['avg_ET3'],a['ET3_obs'],'o', color="blue",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(c['avg_ET3'],c['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['avg_ET3'],d['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3'],e['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3'],f['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
ax.plot(b['avg_ET3'],b['ET3_obs'],'o', color="blue",markersize=2,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_title("RF Day 3 ET Forecast (Irrigated)",size=7)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=8)
ax.locator_params(axis='x', nbins=8)

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
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

a=df[(df['irr_nonirr']==1)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 2386 non irrigated
r2_score(obs,pre) ##  0.70
sqrt(mean_absolute_error(obs,pre))  #21.2
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias   #-2.0

#################################################################################################
### deicide days with extreme events


os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\forecast_results")

df = pd.read_excel('L1_ann.xlsx', sheetname='forc_test')

def crop(df):
    if (df['crop'] =='corn'):
        return 'corn'
    elif (df['crop'] =='soy'):
        return 'soy'
    elif (df['crop'] =='potat1'):
        return 'potat'
    elif (df['crop'] =='potat2'):
        return 'potat'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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

def irrig_non(df):
    if (df['ID'] =='NB_Ne2')or (df['ID'] =='MI_irrig')or (df['ID'] =='WI_CS')or (df['ID'] =='NB_Ne1'):
        return 'irrigated'
    else: 
        return 'non-irrigated'  
    
 
################################################################################################
### make sure to run both RF and LSTM. first we hav RF then we have LSTM
###############################################################################################   
    
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\forecast_results")
df = pd.read_excel('L1_rf.xlsx', sheetname='forc_test')
df_test_rf_L1=df
df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df=df_test_rf_L1

df = pd.read_excel('L2_rf.xlsx', sheetname='forc_test')
df_test_rf_L2=df
df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df_test_rf_L2=df

df = pd.read_excel('L3_rf.xlsx', sheetname='forc_test')
df_test_rf_L3=df
df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df_test_rf_L3=df

##############################################################################################


os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\forecast_results")
df = pd.read_excel('L1_ann.xlsx', sheetname='forc_test')
df_test_rnn_L1=df
df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df=df_test_rnn_L1

df = pd.read_excel('L2_ann.xlsx', sheetname='forc_test')
df_test_rnn_L2=df
df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df_test_rnn_L2=df

df = pd.read_excel('L3_ann.xlsx', sheetname='forc_test')
df_test_rnn_L3=df
df['soil'] = df.apply(soil_df,axis=1)
df['state']=df.apply(state_df,axis=1)
df['irr_nonirr']=df.apply(irrig_non,axis=1)
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df_test_rnn_L3=df

##############################################################################################
## select wet dry months"

#df=df_test_rnn_L1
df=df_test_rf_L1

df["ET1avg"]=(df["rfL1ET1cntr"]+df["rfL1ET1per1"]+df["rfL1ET1per2"]+df["rfL1ET1per3"]+df["rfL1ET1per4"]
+df["rfL1ET1per5"]+df["rfL1ET1per6"]+df["rfL1ET1per7"]+df["rfL1ET1per8"]+df["rfL1ET1per9"]
+df["rfL1ET1per10"])/11


df_std=pd.concat((df["rfL1ET1cntr"],df["rfL1ET1per1"],df["rfL1ET1per2"],df["rfL1ET1per3"]
,df["rfL1ET1per4"],df["rfL1ET1per5"],df["rfL1ET1per6"],df["rfL1ET1per7"],df["rfL1ET1per8"],df["rfL1ET1per9"]
,df["rfL1ET1per10"]),axis=1)

df["ET1_std"]=df_std.std(axis=1)
bac=pd.concat((df['ID'],df['state'],df['TIMESTAMP'], df['month'],df['crop'],df['ET1_obs'],df["ET1avg"],df["ET1_std"]),axis=1)

df=df_test_rf_L2
#df=df_test_rnn_L2

df["ET2avg"]=(df["rfL2ET2cntr"]+df["rfL2ET2per1"]+df["rfL2ET2per2"]+df["rfL2ET2per3"]+df["rfL2ET2per4"]
+df["rfL2ET2per5"]+df["rfL2ET2per6"]+df["rfL2ET2per7"]+df["rfL2ET2per8"]+df["rfL2ET2per9"]
+df["rfL2ET2per10"])/11

df_std=pd.concat((df["rfL2ET2cntr"],df["rfL2ET2per1"],df["rfL2ET2per2"],df["rfL2ET2per3"]
,df["rfL2ET2per4"],df["rfL2ET2per5"],df["rfL2ET2per6"],df["rfL2ET2per7"],df["rfL2ET2per8"],df["rfL2ET2per9"]
,df["rfL2ET2per10"]),axis=1)

df["ET2_std"]=df_std.std(axis=1)
bac["ET2_obs"]=df["ET2_obs"]
bac["ET2avg"]=df["ET2avg"]
bac["ET2_std"]=df["ET2_std"]


#df=df_test_rnn_L3
df=df_test_rf_L3

df["ET3avg"]=(df["rfL3ET3cntr"]+df["rfL3ET3per1"]+df["rfL3ET3per2"]+df["rfL3ET3per3"]+df["rfL3ET3per4"]
+df["rfL3ET3per5"]+df["rfL3ET3per6"]+df["rfL3ET3per7"]+df["rfL3ET3per8"]+df["rfL3ET3per9"]
+df["rfL3ET3per10"])/11

df_std=pd.concat((df["rfL3ET3cntr"],df["rfL3ET3per1"],df["rfL3ET3per2"],df["rfL3ET3per3"]
,df["rfL3ET3per4"],df["rfL3ET3per5"],df["rfL3ET3per6"],df["rfL3ET3per7"],df["rfL3ET3per8"],df["rfL3ET3per9"]
,df["rfL3ET3per10"]),axis=1)

df["ET3_std"]=df_std.std(axis=1)

bac["ET3_obs"]=df["ET3_obs"]
bac["ET3avg"]=df["ET3avg"]
bac["ET3_std"]=df["ET3_std"]

back_rf=bac

#back_rnn=bac
###############################################################################################

### dry month
#array(['MI_T3', 'NB_Ne2', 'MN_Ro3', 'IA_Br1', 'WI_CS', 'MN_Ro1'],
### graph 1
df=back_rf
#df=back_rnn

crit1 = df['TIMESTAMP'].map(lambda x : x.year == 2017)
#crit2 = df['TIMESTAMP'].map(lambda x : x.month == 7)
#new=df[crit1 & crit2]
new=df[crit1]
df=new
#mask = (df['ID'] =="MN_Ro1")
#mask = (df['ID'] =="NB_Ne2")
mask = (df['ID'] =="MI_T3")
#mask = (df['ID'] =="IA_Br1")
df = df.loc[mask]
s=df
m_rf=df.groupby( 'month').mean()
#m_rnn=df.groupby( 'month').mean()

x=[4,5,6,7,8,9,10]
##########################################################################################3
### spread based on ensemble-avg sim
##############################################################################################
########################################################################################3
plt.rcParams['figure.figsize'] = (3,2.2)
fig, ax = plt.subplots()

ax.plot(x,m_rf["ET3_obs"], 'r-',label="Observed")
ax.plot(x, m_rf["ET3avg"], 'o-', color='tab:brown',label="RF_Model")
#ax.fill_between(x, m_rf["ET3avg"]-m_rf["ET3_std"], m_rf["ET3avg"]+m_rf["ET3_std"],alpha=0.4,label="Ensembles SD")

#ax.plot(x, m_rnn["ET3avg"], 'b-', color='tab:green',label="Model")
#ax.fill_between(x, m_rnn["ET3avg"]-m_rnn["ET3_std"], m_rnn["ET3avg"]+m_rnn["ET3_std"],alpha=0.4,label="Ensembles SD")

#ax.plot(s["TIMESTAMP"], s["ET1_obs"], 'r--',marker='o',markersize=1,linewidth=0,label="Observed ET")
#ax.plot(s["TIMESTAMP"], s["ET1avg"], 'k-',label="Day 3 Model Forecast Average")
#ax.fill_between(s["TIMESTAMP"],s["ET1avg"]-s["ET1_std"], s["ET1avg"]+s["ET1_std"],alpha=0.4,label="Ensembles Standard Deviation")


#ax.plot(x, s["ET1_obs"], 'r-',label="Observed ET")
#ax.plot(x, s["ET1avg"], 'k-',label="Day 3 Model Forecast Average")
#ax.fill_between(x,s["ET1avg"]-s["ET1_std"], s["ET1avg"]+s["ET1_std"],alpha=0.4,label="Ensembles Standard Deviation")

#ax.set_title("Corn",size=10)
ax.set_ylabel('Day 3 ET Forecast (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Months')  # we already handled the x-label with ax1

#ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("RF Forecast Model",size=8)
#ax.set_title("RF Forecast Model (US-Ne2, Year 2012, Corn)",size=8)
ax.set_title("RF Forecast Model (US-KM1, Year 2017, Corn)",size=8)

#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
#ax.set_ylim(0,5)
ax.locator_params(axis='x', nbins=100)
plt.xticks([4, 5, 6,7,8,9,10], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
#Formatter = mpl.dates.DateFormatter('%d-%m-%y')

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
#ax.locator_params(axis='y', nbins=5)

ax.set_ylim(0,8)
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

###########################################################################################
## graph 2
#s=df.groupby( 'month').mean()
#x=[4,5,6,7,8,9,10]
s_rf=s
plt.rcParams['figure.figsize'] = (3,2.2)
fig, ax = plt.subplots()
#ax.plot(x,s["ET3_obs"], 'r-',label="Observed")
#ax.plot(x, s["ET3avg"], 'o-', color='tab:brown',label="Model")
#ax.fill_between(x, s["ET3avg"]-s["ET3_std"], s["ET3avg"]+s["ET3_std"],alpha=0.4,label="Ensembles SD")

ax.plot(s_rf["TIMESTAMP"], s_rf["ET3_obs"], 'r--',marker='o',markersize=1,linewidth=0,label="Observed")
ax.plot(s_rf["TIMESTAMP"], s_rf["ET3avg"], 'k-',label="RF_Model")


#ax.fill_between(s_rf["TIMESTAMP"],s_rf["ET3avg"]-s_rf["ET3_std"], s_rf["ET3avg"]+s_rf["ET3_std"],alpha=0.4,label="RF Ensembles SD")

#ax.plot(s_rf["TIMESTAMP"], s_rf["ET3avg"], 'b-',label="LSTM_Model")
#ax.fill_between(s_rnn["TIMESTAMP"],s_rnn["ET3avg"]-s_rnn["ET3_std"], s_rnn["ET3avg"]+s_rnn["ET3_std"],alpha=0.4,label="LSTM Ensembles SD")

#ax.plot(x, s["ET1_obs"], 'r-',label="Observed ET")
#ax.plot(x, s["ET1avg"], 'k-',label="Day 3 Model Forecast Average")
#ax.fill_between(x,s["ET1avg"]-s["ET1_std"], s["ET1avg"]+s["ET1_std"],alpha=0.4,label="Ensembles Standard Deviation")

#ax.set_title("Corn",size=10)
ax.set_ylabel('Day 3 ET Forecast (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Day/Month')  # we already handled the x-label with ax1

#ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.legend(loc='upper right',prop={'size': 6})
ax.set_title("RF Forecast Model",size=8)
#ax.set_title("RF Forecast Model (US-KM1, Year 2017, Corn)",size=8)
#ax.set_title("RF Forecast Model (US-Ne2, Year 2012, Corn)",size=8)
plt.xticks([4, 5, 6,7,8,9,10], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
#ax.set_ylim(0,5)
#ax.locator_params(axis='x', nbins=100)
#plt.xticks([4, 5, 6,7,8,9,10], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
#plt.xticks( ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
#plt.xticks([4, 5, 6,7,8,9,10])
#Formatter = mpl.dates.DateFormatter('%d-%m-%y')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
#plt.xticks([4, 5, 6,7,8,9,10], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

#ax.locator_params(axis='x', nbins=5)

ax.set_ylim(0,8)
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

##############################################################################################

crit1 = df['TIMESTAMP'].map(lambda x : x.year == 2017)
crit2 = df['TIMESTAMP'].map(lambda x : x.month == 6)
#new=df[crit1 & crit2]
new=df[crit1]
df=new
#mask = (df['ID'] =="MN_Ro1")
#mask = (df['ID'] =="NB_Ne2")
mask = (df['ID'] =="MI_T3")
#mask = (df['ID'] =="IA_Br1")
df = df.loc[mask]
s=df

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1\forecast_results")
df = pd.read_excel('forcast_visual.xlsx', sheetname='Sheet1')


plt.rcParams['figure.figsize'] = (3,2.2)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily ET (mm)', color='black')
#ax1.set_xlabel('Day', color='black')

#### use below for Ne2 2012
#ax1.plot(df.iloc[:,0],df.iloc[:,2], color='blue',linestyle="--",label="Previous Days ET",marker='8',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,3], color='red',label="Obs ET",marker='o',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,4], color='black',label="ET Forecast (RF)",marker='o',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,5], color='purple',label="ET Forecast (LSTM)",marker='o',markersize=3)
######## use below MI-T3

ax1.plot(df.iloc[:,0],df.iloc[:,7], color='blue',linestyle="--",label="Previous Days ET",marker='8',markersize=3)
ax1.plot(df.iloc[:,0],df.iloc[:,8], color='red',label="Obs ET",marker='o',markersize=3)
ax1.plot(df.iloc[:,0],df.iloc[:,9], color='black',label="ET Forecast (RF)",marker='o',markersize=3)
ax1.plot(df.iloc[:,0],df.iloc[:,10], color='purple',label="ET Forecast (LSTM)",marker='o',markersize=3)

#ax1.plot(df.iloc[:,0],df.iloc[:,11], color='blue',linestyle="--",label="Previous Days ET",marker='8',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,12], color='red',label="Actual ET",marker='o',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,13], color='black',label="ET Forecast (RF)",marker='o',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,14], color='purple',label="ET Forecast (RNN)",marker='o',markersize=3)


#ax1.plot(df.iloc[:,0],df.iloc[:,15], color='blue',linestyle="--",label="Previous Days ET",marker='8',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,17], color='red',label="Actual ET",marker='o',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,18], color='black',label="ET Forecast (RF)",marker='o',markersize=3)
#ax1.plot(df.iloc[:,0],df.iloc[:,19], color='purple',label="ET Forecast (RNN)",marker='o',markersize=3)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')

ax1.axvline(x=4, color='green', linestyle='--', lw=2)

ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='lower left',fontsize="6")

ax1.yaxis.label.set_size(10)
ax1.xaxis.label.set_size(10) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 10)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax1.set_ylim(0,8)
plt.xticks([1, 2, 3,4,5,6], [ '-3', '-2','-1','Day1','Day2','Day3'])
#Formatter = mpl.dates.DateFormatter('%d-%m-%y')

#ax1.set_xlim(min(df.iloc[:,0]),max(df.iloc[:,0]+4))
#ax1.set_title("5/21/2017-5/26/2017 (MN-Soybean silt Loam)",fontsize=12)

#ax1.set_title("8/27/2017-9/21/2017 (MI-Corn sandy Loam)",fontsize=12)

#ax1.set_title("7/22/2018-7/27/2018 (WI-Potato Loamy sand)",fontsize=12)
#ax1.set_title("6/12 to 6/17/2012 (US-Ne2, Year 2012,Silt Loam, Corn)",fontsize=6)
ax1.set_title("7/4 to 7/9/2017 (US-KM1, Year 2017, Sandy Loam Corn)",fontsize=6)

#myFmt = mdates.DateFormatter('%Y-%m')
#ax1.xaxis.set_major_formatter(myFmt)
ax1 = plt.gca()
#ax.invert_yaxis()
fig.autofmt_xdate() 

plt.show()
#fig.savefig('vector4.png',dpi=500,bbox_inches = "tight")
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()









