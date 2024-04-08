# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:37:37 2020

@author: Ammara
"""


import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from math import*
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble.forest import _generate_unsampled_indices
#from rfpimp import *
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from math import sqrt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import roc_auc_score
import scipy
from scipy.stats import mode

def mode_(s):
    try:
        return s.mode()[0]
   
    except IndexError:
        return np.nan
os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata")

comp=pd.read_csv("complte_data_2019.csv")
df1=comp
cols = df.columns.tolist()


os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata\NLDAS")
df=pd.read_csv('NOAA_1.csv')
df.index = pd.DatetimeIndex(df.DateTime)
back1=df
S1=back1
os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata\NLDAS")
df=pd.read_csv('NOAA_2.csv')
df.index = pd.DatetimeIndex(df.DateTime)
back2=df

S2=back2

set1=S1.groupby(["ID"]).resample('D').sum().reset_index()
set2=S2.groupby(["ID"]).resample('D').sum().reset_index()

#S2 = back2[back2['ID'].isnull()]
reframed=pd.concat((set1,set2),axis=0)
df2=reframed

df2= df2[df2['NOAA'] != 0]

backup=df2
df=backup
## total number should be 26331
df=df.set_index('ID')   
df.index.unique()


# may b make id as index   

df1=df.loc[['NB_Ne1','MI_nonirri','IL_Bo1','MN_Ro5','IL_B02','MI_irrig','OH_CRT','MN_Ro2','MI_T4','NB_Ne3','IA_Br3']]

df2=df.loc[['MN_Ro6','IL_1B1','MI_T3','NB_Ne2','MN_Ro3','IA_Br1','WI_CS','MN_Ro1']]


#df.loc[['MN_Ro2']]

#df.loc[['WI_CS']]

df=pd.concat((df1,df2),axis=0)

df.reset_index(level=0, inplace=True)

df.index = np.arange(1, len(df) + 1)

df.drop(['NOAA_new'], axis=1, inplace=True)

#only summer month
list_months=[4,5,6,7,8,9,10]
df=df[pd.to_datetime(df['DateTime']).dt.month.isin(list_months)]
summ=df

summ.to_csv('summ_NLDAS.csv', index=False, header=True)

#########################################################################
# remove all rows where ET is zero
#########################################################################

df = df[pd.notnull(df['ET'])]
summ=df

# add id to summer data
df['idn'] = np.arange(len(df))




















df2.reindex(['IL_1B1', 'IL_B02', 'NB_Ne2', 'NB_Ne3', 'WI_CS', 'IA_Br1',
       'MN_Ro1', 'MN_Ro2', 'MN_Ro3', 'MN_Ro5', 'MI_T4', 'MI_nonirri',
       'OH_CRT', 'MI_T3', 'MI_irrig', 'MN_Ro6', 'NB_Ne1', 'IA_Br3',
       'IL_Bo1'])


df.reindex(["Z", "C", "A"])

array(['IL_1B1', 'IL_B02', 'NB_Ne2', 'NB_Ne3', 'WI_CS', 'IA_Br1',
       'MN_Ro1', 'MN_Ro2', 'MN_Ro3', 'MN_Ro5', 'MI_T4', 'MI_nonirri',
       'OH_CRT', 'MI_T3', 'MI_irrig', 'MN_Ro6', 'NB_Ne1', 'IA_Br3',
       'IL_Bo1'], dtype=object)



df2 = df2.loc[['IL_1B1', 'IL_B02', 'NB_Ne2', 'NB_Ne3', 'WI_CS', 'IA_Br1',
       'MN_Ro1', 'MN_Ro2', 'MN_Ro3', 'MN_Ro5', 'MI_T4', 'MI_nonirri',
       'OH_CRT', 'MI_T3', 'MI_irrig', 'MN_Ro6', 'NB_Ne1', 'IA_Br3',
       'IL_Bo1']]



#df2_IL_1B1=df2[df2['ID'].str.contains("MN_Ro2")]
#df1_IL_1B1=df1[df1['ID'].str.contains("MN_Ro2")]


df=reframed.loc[['IL_1B1','IL_B02','NB_Ne2','NB_Ne3','WI_CS','IA_Br1','MN_Ro1',
 ### checked above it                



                 
'MN_Ro2','MN_Ro3','MN_Ro5','MI_T4','MI_nonirri','OH_CRT','MI_T3','MI_irrig',
'MN_Ro6','NB_Ne1','IA_Br3','IL_Bo1']]

os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata\NLDAS")
df=pd.read_csv('NOAA_2.csv')
df.index = pd.DatetimeIndex(df.DateTime)
back2=df
df.ID.unique()


S2 = df[df['ID'].isnull()]

reframed=pd.concat((S1,S2),axis=0)
df=reframed

#df=pd.concat((reframed["DateTime"],reframed["ID"],reframed["NOAA_new"]),axis=1)
df["NLDAS_new"]=df["NLDAS_new"]*3600

set1=df.groupby(["ID"]).resample('D').sum().reset_index()
set1["NLDAS_new"]=set1["NLDAS_new"]


### pick on by one 


df1=df[df['ID'].str.contains("IL_1B1")]




df=reframed.loc[['IL_1B1','IL_B02','NB_Ne2','NB_Ne3','WI_CS','IA_Br1','MN_Ro1','MN_Ro2','MN_Ro3','MN_Ro5','MI_T4','MI_nonirri','OH_CRT','MI_T3','MI_irrig','MN_Ro6','NB_Ne1','IA_Br3','IL_Bo1']]

##################################################################################################

os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata\NLDAS")
df=pd.read_csv('GLDAS.csv')
df.index = pd.DatetimeIndex(df.DateTime)
back1=df

df=back1

df["GLDAS1_new"]=df["GLDAS1_new"]*3600
#df.ID.unique()

#S1 = df.groupby(['ID']).resample('D',  fill_method='ffill')

#df=df.drop(['DateTime'], axis=1)

set1=df.groupby(["ID"]).resample('D').sum().reset_index()
set1["GLDAS1_new"]=set1["GLDAS1_new"]

#df1=set1[set1['ID'].str.contains("IL_1B1")]


df1=set1[set1['ID'].str.contains("IA_Br1")]




os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata\NLDAS")
df=pd.read_csv('GLDAS.csv')
df.index = pd.DatetimeIndex(df.DateTime)
back2=df

df.ID.unique()

#NOAA2_=df.resample('D', how='sum')
S2 = df.groupby(['ID']).resample('D',  fill_method='ffill')

#S2 = df[df['ID'].isnull()]

reframed=pd.concat((S1,S2),axis=0)

df=pd.concat((reframed["DateTime"],reframed["ID"],reframed["NOAA_new"]),axis=1)


### pick on by one 

df1=df[df['ID'].str.contains("IL_1B1")]


df=reframed.loc[['IL_1B1','IL_B02','NB_Ne2','NB_Ne3','WI_CS','IA_Br1','MN_Ro1','MN_Ro2','MN_Ro3','MN_Ro5','MI_T4','MI_nonirri','OH_CRT','MI_T3','MI_irrig','MN_Ro6','NB_Ne1','IA_Br3','IL_Bo1']]
























