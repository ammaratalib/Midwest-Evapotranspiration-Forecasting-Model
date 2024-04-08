# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:02:25 2020

@author: Ammara
"""

#https://seaborn.pydata.org/generated/seaborn.lineplot.html
import itertools 
import chart_studio.plotly
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
df=df_test_rf_L1

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

df=df_test_rf_L2

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

#### try ET day 3 Level 3

df=df_test_rf_L3

df["avg"]=(df["rfL3ET3cntr"]+df["rfL3ET3per1"]+df["rfL3ET3per2"]+df["rfL3ET3per3"]+df["rfL3ET3per4"]
+df["rfL3ET3per5"]+df["rfL3ET3per6"]+df["rfL3ET3per7"]+df["rfL3ET3per8"]+df["rfL3ET3per9"]
+df["rfL3ET3per10"])/11

#df["std"]=(df["rfL1ET1cntr"]-df["ET1_obs"]).std()

#############################################################################################
### resid  ensem-obs
############################################################################################

df["resid"]=((df["rfL3ET3cntr"]-df["ET3_obs"])+(df["rfL3ET3per1"]-df["ET3_obs"])+(df["rfL3ET3cntr"]-df["ET3_obs"])
+(df["rfL3ET3per1"]-df["ET3_obs"])+(df["rfL3ET3per2"]-df["ET3_obs"])+(df["rfL3ET3per3"]-df["ET3_obs"])
+(df["rfL3ET3per4"]-df["ET3_obs"])+(df["rfL3ET3per5"]-df["ET3_obs"])+(df["rfL3ET3per6"]-df["ET3_obs"])
+(df["rfL3ET3per7"]-df["ET3_obs"])+(df["rfL3ET3per8"]-df["ET3_obs"])+(df["rfL3ET3per9"]-df["ET3_obs"])
+(df["rfL3ET3per10"]-df["ET3_obs"]))/11

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






















+df["rfL1ET1per1"]+df["rfL1ET1per2"]+df["rfL1ET1per3"]+df["rfL1ET1per4"]
+df["rfL1ET1per5"]+df["rfL1ET1per6"]+df["rfL1ET1per7"]+df["rfL1ET1per8"]+df["rfL1ET1per9"]
+df["rfL1ET1per10"])/11


error=simulated-observed
Do the same for all perturbation and then calculate an average error. Then use that average error in the figure for an ensemble simulated ET. 
plot (month, simulated+error,simulated+error)
So basically fill the space between    simulated+error and  simulated+error  with red color.


plot (month, simulated+error,simulated+error)




df=df_test_rf_L1
def avg( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = (g['rfL1ET1cntr']+g['rfL1ET1per1']+g['rfL1ET1per2']+g['rfL1ET1per3']+g['rfL1ET1per4']
    +g['rfL1ET1per5']+g['rfL1ET1per6']+g['rfL1ET1per7']+g['rfL1ET1per8']+g['rfL1ET1per9']+g['rfL1ET1per10'])/11    
     rmsecntr= rmsecntr*25.4
    
    #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return (rmsecntr)
s=df.groupby( 'month' ).apply(resid)



a = np.array([[1, 2], [3, 4]])
np.std(a)
1.1180339887498949 # may vary









df=df_test_rf_L1
def resid( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = ((g['ET1_obs']-g['rfL1ET1cntr'])*25.4)
    rmseper1=((g['ET1_obs']-g['rfL1ET1per1'])*25.4)
    rmseper2=((g['ET1_obs']-g['rfL1ET1per2'])*25.4)
    rmseper3=((g['ET1_obs']-g['rfL1ET1per3'])*25.4)
    rmseper4=((g['ET1_obs']-g['rfL1ET1per4'])*25.4)
    rmseper5=((g['ET1_obs']-g['rfL1ET1per5'])*25.4)
    rmseper6=((g['ET1_obs']-g['rfL1ET1per6'])*25.4)
    rmseper7=((g['ET1_obs']-g['rfL1ET1per7'])*25.4)
    rmseper8=((g['ET1_obs']-g['rfL1ET1per8'])*25.4)
    rmseper9=((g['ET1_obs']-g['rfL1ET1per9'])*25.4)
    rmseper10=((g['ET1_obs']-g['rfL1ET1per10'])*25.4)
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return (rmsecntr,rmseper1,rmseper2,rmseper3,rmseper4,rmseper5,rmseper6,rmseper7,rmseper8,rmseper9,rmseper10)
s=df.groupby( 'month' ).apply(resid)















df=df_test_rf_L1
def ET1_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = ( mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1cntr']*25.4 ) )
    rmseper1 = (mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per1']*25.4 ) )
    rmseper2 = (mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per2']*25.4 ) )
    rmseper3 = (mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per3']*25.4 ) )
    rmseper4 = ( mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per4']*25.4 ) )
    rmseper5 = (mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per5']*25.4 ) )
    rmseper6 = ( mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per6']*25.4 ) )
    rmseper7 = (mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per7']*25.4 ) )
    rmseper8 = (mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per8']*25.4 ) )
    rmseper9 = ( mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per9']*25.4 ) )
    rmseper10 =(mean_absolute_error(g['ET1_obs']*25.4, g['rfL1ET1per10']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return (rmsecntr,rmseper1,rmseper2,rmseper3,rmseper4,rmseper5,rmseper6,rmseper7,rmseper8,rmseper9,rmseper10)
s=df.groupby( 'month' ).apply(ET1_rmse)


def ET2_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2cntr']*25.4 ) )
    rmseper1 = (  mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per1']*25.4 ) )
    rmseper2 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per2']*25.4 ) )
    rmseper3 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per3']*25.4 ) )
    rmseper4 = (  mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per4']*25.4 ) )
    rmseper5 = (  mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per5']*25.4 ) )
    rmseper6 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per6']*25.4 ) )
    rmseper7 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per7']*25.4 ) )
    rmseper8 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per8']*25.4 ) )
    rmseper9 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per9']*25.4 ) )
    rmseper10 = ( mean_absolute_error(g['ET2_obs']*25.4, g['rfL1ET2per10']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return ([rmsecntr,rmseper1,rmseper2,rmseper3,rmseper4,rmseper5,rmseper6,rmseper7,rmseper8,rmseper9,rmseper10])

s=df.groupby( 'month' ).apply(ET2_rmse)



def ET3_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmsecntr = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3cntr']*25.4 ) )
    rmseper1 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per1']*25.4 ) )
    rmseper2 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per2']*25.4 ) )
    rmseper3 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per3']*25.4 ) )
    rmseper4 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per4']*25.4 ) )
    rmseper5 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per5']*25.4 ) )
    rmseper6 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per6']*25.4 ) )
    rmseper7 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per7']*25.4 ) )
    rmseper8 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per8']*25.4 ) )
    rmseper9 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per9']*25.4 ) )
    rmseper10 = ( mean_absolute_error(g['ET3_obs']*25.4, g['rfL1ET3per10']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return ([rmsecntr,rmseper1,rmseper2,rmseper3,rmseper4,rmseper5,rmseper6,rmseper7,rmseper8,rmseper9,rmseper10])

s=df.groupby( 'month' ).apply(ET3_rmse)



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
df["residET1"]=(df["avg_ET1"]-df["ET1_obs"])*25.4
df_ET1=df
df=df_test_rf_L2
df['soil'] = df.apply(soil_df,axis=1)
df['crop'] = df.apply(crop,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET2"]=(df['rfL2ET2cntr'])
df["residET2"]=(df["avg_ET2"]-df["ET2_obs"])*25.4
df_ET2=df

df=df_test_rf_L3
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET3"]=(df['rfL3ET3cntr'])
df["residET3"]=(df["avg_ET3"]-df["ET3_obs"])*25.4
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

plt.rcParams['figure.figsize'] = (2.0, 1.5)
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
ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
ax1.set_ylim(-200,200)

ax1.set_title("RF ET day 3 Forecasting",fontsize=6)
ax1.set_xticklabels( ('Corn','Potatoes','Corn','Soybeans','Soybeans','Corn'))
ax1.set_xticks(np.arange(1,7))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig2.png',dpi=300, bbox_inches = "tight")
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
df["avg_ET2"]=(df['rfL1ET2cntr'])
df["residET2"]=(df["avg_ET2"]-df["ET2_obs"])
df_ET2=df

df=df_test_rnn_L3
df['crop'] = df.apply(crop,axis=1)
df['soil'] = df.apply(soil_df,axis=1)
df = df[pd.notnull(df['crop'])]
df = df[df.crop != '0']
df["avg_ET3"]=(df['rfL1ET3cntr'])
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

plt.rcParams['figure.figsize'] = (2.0, 1.5)
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
ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
ax1.set_ylim(-200,200)
plt.legend(*zip(*labels),loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 4.5})
ax1.set_title("LSTM ET day 3 Forecasting ",fontsize=6)
ax1.set_xticklabels( ('Corn','Potatoes','Corn','Soybeans','Soybeans','Corn'))
ax1.set_xticks(np.arange(1,7))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig2.png',dpi=300, bbox_inches = "tight")
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
ax.plot(a['avg_ET3']*25.4,a['ET3_obs']*25.4,'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(c['avg_ET3']*25.4,c['ET3_obs']*25.4,'p', color="seagreen",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['avg_ET3']*25.4,d['ET3_obs']*25.4,'s', color="gold",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3']*25.4,e['ET3_obs']*25.4,'h', color="red",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3']*25.4,f['ET3_obs']*25.4,'v', color="brown",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
#ax.plot(b['avg_ET3']*25.4,b['ET3_obs']*25.4,'>', color="black",markersize=,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend()
lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})
ax.set_title("LSTM Day 3 ET Forecast (Non Irrigated)",size=7)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,250)
ax.set_xlim(0,250)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

a=df[(df['irr_nonirr']==0)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 5347 non irrigated
r2_score(obs,pre) ##  0.47
sqrt(mean_absolute_error(obs,pre))*25.4  #23.2
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
ax.plot(e['avg_ET3']*25.4,e['ET3_obs']*25.4,'h', color="red",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3']*25.4,f['ET3_obs']*25.4,'v', color="brown",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
ax.plot(b['avg_ET3']*25.4,b['ET3_obs']*25.4,'>', color="black",markersize=2,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_title("LSTM Day 3 ET Forecast (Irrigated)",size=7)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.legend()
lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6})

ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,250)
ax.set_xlim(0,250)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

a=df[(df['irr_nonirr']==1)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 2386  irrigated
r2_score(obs,pre) ##  0.67
sqrt(mean_absolute_error(obs,pre))*25.4  #21.9
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
ax.plot(a['avg_ET3']*25.4,a['ET3_obs']*25.4,'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(c['avg_ET3']*25.4,c['ET3_obs']*25.4,'p', color="seagreen",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['avg_ET3']*25.4,d['ET3_obs']*25.4,'s', color="gold",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3']*25.4,e['ET3_obs']*25.4,'h', color="red",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3']*25.4,f['ET3_obs']*25.4,'v', color="brown",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")

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
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,250)
ax.set_xlim(0,250)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

a=df[(df['irr_nonirr']==0)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 2386 non irrigated
r2_score(obs,pre) ##  0.53
sqrt(mean_absolute_error(obs,pre))*25.4  #22.4
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
ax.plot(a['avg_ET3']*25.4,a['ET3_obs']*25.4,'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(c['avg_ET3']*25.4,c['ET3_obs']*25.4,'p', color="seagreen",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['avg_ET3']*25.4,d['ET3_obs']*25.4,'s', color="gold",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3']*25.4,e['ET3_obs']*25.4,'h', color="red",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3']*25.4,f['ET3_obs']*25.4,'v', color="brown",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
ax.plot(b['avg_ET3']*25.4,b['ET3_obs']*25.4,'>', color="black",markersize=2,fillstyle='none',label="Potatoes (Loamy Sand)")

ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_title("RF Day 3 ET Forecast (Irrigated)",size=7)
ax.set_ylabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted ET (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)

ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,250)
ax.set_xlim(0,250)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

a=df[(df['irr_nonirr']==1)]
pre=a['avg_ET3']
obs=a['ET3_obs']

#scipy.stats.pearsonr(df.ET1_obs,df.rnnL1ET3)
# sample size 2386 non irrigated
r2_score(obs,pre) ##  0.70
sqrt(mean_absolute_error(obs,pre))*25.4  #21.2
pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias   #-2.0

#################################################################################################