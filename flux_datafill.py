# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:03:40 2019

@author: ammara
"""
reset

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from math import*
###############################################################################
##  per min flux data
###############################################################################
#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\illinois\US_1B1\US-IB1")
#df=pd.read_csv("US-IB1.csv") 

###########################################
#USIB1
# from 2006 to 2017
#210384
#4383
##########################################
#US_Bo1_combine
# from 1998 to 2017   2009 is missing 
#333120
#6940
#########################################################################
#########################################################################

#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\nebraska\US-Ne2")

os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\ohio\AMF_US-CRT_BASE-BADM_3-5")

df=pd.read_csv("US-CRT.csv")
#2005  2011

df.head(10)

df.tail(10)

idx = pd.date_range(start='01/1/2011', periods=52608, freq='30T')
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 

df.max(axis = 0) 

df.min(axis = 0)

# Threshold to remove outliers 

# usually LE >800 is seen in deserts 
# LE <-200 and >800
# H <-200 and >800
# NEE<-50 and >50
#Rg <-50 and >1200
#ustar <0 and >1000
#tair <-200 and >200 in celcius 
#rH >0 and  <100
#vpd<0 and  >100 


df["LE1"] = np.where(df["LE"]<-200, np.NaN, df["LE"])
df["LE2"] = np.where(df["LE1"]>800, np.NaN, df["LE1"])
df["LE"]=df["LE2"]


df["H1"] = np.where(df["H"]<-200, np.NaN, df["H"])
df["H2"] = np.where(df["H1"]>800, np.NaN, df["H1"])
df["H"]=df["H2"]



df["NEE1"] = np.where(df["NEE"]<-50, np.NaN, df["NEE"])
df["NEE2"] = np.where(df["NEE1"]>50, np.NaN, df["NEE1"])
df["NEE"]=df["NEE2"]

df["Rnet1"] = np.where(df["Rnet"]<-500, np.NaN, df["Rnet"])
df["Rnet2"] = np.where(df["Rnet1"]>1000, np.NaN, df["Rnet1"])
df["Rnet"]=df["Rnet2"]

df["Rg1"] = np.where(df["Rg"]<-50, np.NaN, df["Rg"])
df["Rg2"] = np.where(df["Rg1"]>1200, np.NaN, df["Rg1"])
df["Rg"]=df["Rg2"]

df["Ustar1"] = np.where(df["Ustar"]<0, np.NaN, df["Ustar"])
df["Ustar2"] = np.where(df["Ustar1"]>3, np.NaN, df["Ustar1"])
df["Ustar"]=df["Ustar2"]

df["Tair1"] = np.where(df["Tair"]<-200, np.NaN, df["Tair"])
df["Tair2"] = np.where(df["Tair1"]>200, np.NaN, df["Tair1"])
df["Tair"]=df["Tair2"]


df["rH1"] = np.where(df["rH"]<0, np.NaN, df["rH"])
df["rH2"] = np.where(df["rH1"]>100, np.NaN, df["rH1"])
df["rH"]=df["rH2"]

df["VPD1"] = np.where(df["VPD"]<0, np.NaN, df["VPD"])
df["VPD2"] = np.where(df["VPD1"]>100, np.NaN, df["VPD1"])
df["VPD"]=df["VPD2"]



es=0.6108*np.exp((17.27*df["Tair"])/(df["Tair"]+237.3))
ea=df["rH"]/100*es
VPD=(es-ea)*10
df['VPD']=VPD
 
df.max(axis = 0) 
df.min(axis = 0)

##################################################################### 
# this step is for  consecutive years 
#index is correct date.TIMESTAMP date mess up for missing values
df=df.iloc[:,1:99]  # don't worry about end column. not so important
 # don't worry about end column. not so important
df.columns.tolist() 
df.head()
df.describe()
df=df.apply(pd.to_numeric, errors='coerce')
df=df.astype(float) 
df=df.rename_axis('TIMESTAMP').reset_index() 
df=df.resample('30min', on='TIMESTAMP').mean()
df=df.reset_index()
timestamp=df
# an index column starting with 0 and first column datetime
# ey proc will not work if years are not consecutive
############################################################
df['Year'] = df['TIMESTAMP'].dt.year
#df['day'] = df['TIMESTAMP'].dt.day
df['DoY'] = df['TIMESTAMP'].dt.dayofyear
# eddy proc need  these columns in the start
df['Tsoil']=np.nan
s=pd.Series(np.arange(0.5,24.5,0.5))
############################################################################
k=pd.concat([s]*(4748),axis=0)
k.reset_index(drop=True, inplace=True)
#k.index = np.arange(1, len(k)+1)
df.insert(2,'Hour',k)
df=df.iloc[:,1:103]
# non consecutive years

#df=df.iloc[:,0:103]
df.head(10)
df.tail(10)

#also added a nan column in dataframe
timestamp["DateTime"]=timestamp["TIMESTAMP"]
cols = df.columns.tolist()
reframed=pd.concat((timestamp["DateTime"],df['Year'],df['DoY'],df['Hour'],df["NEE"],df["LE"],df["H"],df['Rg'],df["Tair"],df['Tsoil'],df["rH"],df["VPD"],df["Ustar"]),axis=1)
#reframed=pd.concat((timestamp["DateTime"],df['Year'],df['DoY'],df['Hour'],df["NEE"],df["LE"],df["H"],df["Tair"],df['Tsoil'],df["rH"],df["VPD"],df["Ustar"]),axis=1)

reframed.tail(10)
reframed.to_csv('gaps.csv', index=False, header=True)
#####################################################################
#after gap fill

fill=pd.read_csv("filled.csv") 
fill.insert(0,'TIMESTAMP',timestamp["TIMESTAMP"])

# ET is in mm 
fill["ET"]=fill["LE"]*((1/1000)*(1/(2.5*1000000))*(86400 *1000))
#fill["ET"]=fill["ET"]*0.0393701
fill=fill.resample('D', on='TIMESTAMP').mean()
fill["ET"]= np.where(fill["ET"]<0,0.00001,fill["ET"])
fill=fill.reset_index()

fill.max(axis = 0) 
fill.min(axis = 0) 

obs_ET=pd.concat((fill["TIMESTAMP"],fill["ET"]),axis=1)
obs_ET.to_csv('obs_ET.csv', index=False, header=True)



##########################################################################
# fix precip
#######################################################################
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\iowa\US-Br3")

#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\Wisconsin")

os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\ohio")

df=pd.read_csv("precip_fix.csv")

#2005  2011

df.head(10)

df.tail(10)
# only days
#change this number based on data
idx = pd.date_range(start='01/01/2011', periods=1096,freq='D')

df.index = pd.DatetimeIndex(df.TIMESTAMP)

df=df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values

df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:7]  # don't worry about end column. not so important
df=df.apply(pd.to_numeric, errors='coerce')
df=df.astype(float) 
df=df.rename_axis('TIMESTAMP').reset_index() 
df=df.resample('D', on='TIMESTAMP').mean()
df=df.reset_index()



df.max(axis = 0) 
df.min(axis = 0)

df.to_csv('precip_fix.csv', index=False, header=True)

df.head(10)

df.tail(10)
##########################################################################
# fix radia
#######################################################################

# find max min values

# make sure to change dae formate in excel. excel will give an error. 
# so just simply use idx in python as use it 

#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\illinois\US-Bo2")


#os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\nebraska\US-Ne3")

os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\ohio")

df=pd.read_csv("radia_fix.csv")

df.head(10)
df.tail(10)

idx = pd.date_range(start='01/1/2011', periods= 26304, freq='1H')
#idx = pd.date_range(start='01/1/2018', periods=558, freq='1D')



df.index = pd.DatetimeIndex(df.TIMESTAMP)

df=df.reindex(idx,fill_value=-9999) #add missing dates and add  nan for missing values

df= df.replace(-9999, np.nan, regex=True) 

df=df.iloc[:,1:7]  # don't worry about end column. not so important
 # don't worry about end column. not so important


df=df.apply(pd.to_numeric, errors='coerce')
df=df.astype(float) 
df=df.rename_axis('TIMESTAMP').reset_index() 
df=df.resample('D', on='TIMESTAMP').mean()
df=df.reset_index()

df["wind"]=np.sqrt((df["zonal"]**2)+(df["meridonal"]**2))
reframed=pd.concat((df["TIMESTAMP"],df['LW'],df['SW'],df['wind']),axis=1)
reframed.to_csv('radia_fix.csv', index=False, header=True)

reframed.head(10)

reframed.tail(10)

#######################################################################

# moving average for precipitation

reset

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from math import*



os.chdir(r"G:\ML_ET\random_forest\input_fluxdata\ohio")

df=pd.read_csv("precip_fix.csv")

#df=pd.read_excel('compile.xlsx', sheetname='WCS_precip')

df=df.iloc[:,1]

df=df.fillna(df.rolling(2,1).mean())

rolling_mean = df.rolling(window=7).mean()

rolling_mean = df.rolling(window=15).mean()

rolling_mean = df.rolling(window=30).mean()

rolling_mean = df.rolling(window=60).mean()

#######################################################################
#=IF(AND(M2>86),AVERAGE(86,N2)-50,IF(OR(N2<50,M2<50),AVERAGE(50,50)-50,AVERAGE(M2,N2)-50))

# caclulate GDD 

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from math import*



os.chdir(r"G:\ML_ET\random_forest\input_fluxdata")
# excel sheets take long. Try 
#df=pd.read_excel('compile.xlsx', sheetname='all_data')
df=pd.read_csv("orig_data.csv")
cols = df.columns.tolist()


##### initial code to calculate GDD for all crops

def flag_df(df):    
#########################################################################
    #corn  and soy
##########################################################################
    ### hot day
    ################
    if (df['crop'] =='corn' or df['crop'] =='soy' or df['crop'] =='kura') and (df['tmax_F']>86):
#        return np.mean(86,df['tmin_F'])-50
        return (((86+df['tmin_F'])/2)-50)  
    ### cold day
    #################        
    elif (df['crop'] =='corn' or df['crop'] =='soy'or df['crop'] =='kura') and (df['tmax_F']<50 or df['tmin_F']<50) :
#        return np.mean(86,df['tmin_F'])-50
        return (((50+50)/2)-50)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='corn' or df['crop'] =='soy'or df['crop'] =='kura')  :
#        return np.mean(86,df['tmin_F'])-50
        return (((df['tmax_F']+df['tmin_F'])/2)-50)
########################
#potato
#######################
    ### hot day
    ################
    elif (df['crop'] =='potat1' or df['crop'] =='potat2') and (df['tmax_F']>86):
#        return np.mean(86,df['tmin_F'])-50
        return (((86+df['tmin_F'])/2)-45)  
     ### cold day
    #################        
    elif (df['crop'] =='potat1' or df['crop'] =='potat2') and (df['tmax_F']<45 or df['tmin_F']<45) :
#        return np.mean(86,df['tmin_F'])-50
        return (((45+45)/2)-45)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='potat1' or df['crop'] =='potat2')  :
#        return np.mean(86,df['tmin_F'])-50
        return (((df['tmax_F']+df['tmin_F'])/2)-45)

########################
     # wheat   
########################    
    ### hot day
    ################
    elif (df['crop'] =='wheat') and (df['tmax_F']>70):
#        return np.mean(86,df['tmin_F'])-50
        return (((70+df['tmin_F'])/2)-40)  
     ### cold day
    #################        
    elif (df['crop'] =='wheat') and (df['tmax_F']<40 or df['tmin_F']<40) :
#        return np.mean(86,df['tmin_F'])-50
        return (((40+40)/2)-40)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='wheat')  :
#        return np.mean(86,df['tmin_F'])-50
        return (((df['tmax_F']+df['tmin_F'])/2)-40)
########################
     # winter wheat   
########################    
    ### hot day
    ################
    elif (df['crop'] =='wint_wheat') and (df['tmax_F']>70):
#        return np.mean(86,df['tmin_F'])-50
        return (((70+df['tmin_F'])/2)-32)  
     ### cold day
    #################        
    elif (df['crop'] =='wint_wheat') and (df['tmax_F']<32 or df['tmin_F']<32) :
#        return np.mean(86,df['tmin_F'])-50
        return (((32+32)/2)-32)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='wint_wheat')  :
#        return np.mean(86,df['tmin_F'])-50

        return (((df['tmax_F']+df['tmin_F'])/2)-32)

    else: 
        return '0'   

df['GDD'] = df.apply(flag_df, axis = 1)

df["GDD_abs"]=(df['GDD'].astype(float)).abs()

#backup
df2=df
##### change GDD or abs GDD value for winter wheat to 880. becaue default code
## will not change it    #31330
##### change GDDor abs GDD value for potato1 crop to 809. becaue default code
## will not change it  row #13880

# we are changing GDD_abs column 

df.iat[13880,30] = 809  # potatoe because it was already planted
df.iat[31330,30] = 0   # winter wheat because it was planted in oct

df.iat[31422,30]=927.2# winter wheat 2012 values need to continue for 2013 
#because it was not harvested 

#only need to do for potato. no need to do for winter wheat

df1=pd.concat((df["ID"],df["year"],df["crop"],df["GDD_abs"]),axis=1)

df1['CumGDD'] = df1.groupby(by=["ID","year","crop"])["GDD_abs"].cumsum()

df["CumGDD"]=df1["CumGDD"]
#######################################################################

###  modified GDD code for winter wheat

def flag_df(df):
    
#########################################################################
    #corn  and soy
##########################################################################
    ### hot day
    ################
    if (df['crop'] =='corn' or df['crop'] =='soy' or df['crop'] =='kura') and (df['tmax_F']>86):
#        return np.mean(86,df['tmin_F'])-50
        return (((86+df['tmin_F'])/2)-50)  
    ### cold day
    #################        
    elif (df['crop'] =='corn' or df['crop'] =='soy'or df['crop'] =='kura') and (df['tmax_F']<50 or df['tmin_F']<50) :
#        return np.mean(86,df['tmin_F'])-50
        return (((50+50)/2)-50)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='corn' or df['crop'] =='soy'or df['crop'] =='kura')  :
#        return np.mean(86,df['tmin_F'])-50
        return (((df['tmax_F']+df['tmin_F'])/2)-50)
########################
#potato
#######################
    ### hot day
    ################
    elif (df['crop'] =='potat1' or df['crop'] =='potat2') and (df['tmax_F']>86):
#        return np.mean(86,df['tmin_F'])-50
        return (((86+df['tmin_F'])/2)-45)  
     ### cold day
    #################        
    elif (df['crop'] =='potat1' or df['crop'] =='potat2') and (df['tmax_F']<45 or df['tmin_F']<45) :
#        return np.mean(86,df['tmin_F'])-50
        return (((45+45)/2)-45)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='potat1' or df['crop'] =='potat2')  :
#        return np.mean(86,df['tmin_F'])-50
        return (((df['tmax_F']+df['tmin_F'])/2)-45)

########################
#wheat   
########################    
    ### hot day
    ################
    elif (df['crop'] =='wheat') and (df['tmax_F']>70) and (df['CumGDD']<395) :
#        return np.mean(86,df['tmin_F'])-50
        return (((70+df['tmin_F'])/2)-40)  

#################  
#after a certain maurity
    elif (df['crop'] =='wheat') and (df['tmax_F']>95) and (df['CumGDD']>=395) :
#        return np.mean(86,df['tmin_F'])-50
        return (((95+df['tmin_F'])/2)-40)  
#### cold day     
    elif (df['crop'] =='wheat') and (df['tmax_F']<40 or df['tmin_F']<40) :
#        return np.mean(86,df['tmin_F'])-50
        return (((40+40)/2)-40)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='wheat')  :
#        return np.mean(86,df['tmin_F'])-50
        return (((df['tmax_F']+df['tmin_F'])/2)-40)
########################
     # winter wheat   
########################    
    ### hot day
    ################
    elif (df['crop'] =='wint_wheat') and (df['tmax_F']>70) and (df['CumGDD']<395) :
#        return np.mean(86,df['tmin_F'])-50
        return (((70+df['tmin_F'])/2)-32)  

#################  
#after a certain maurity
    elif (df['crop'] =='wint_wheat') and (df['tmax_F']>95) and (df['CumGDD']>=395) :
#        return np.mean(86,df['tmin_F'])-50
        return (((95+df['tmin_F'])/2)-32)  

#### cold day     
    elif (df['crop'] =='wint_wheat') and (df['tmax_F']<32 or df['tmin_F']<32) :
#        return np.mean(86,df['tmin_F'])-50
        return (((32+32)/2)-32)
    ##################
    #normal day
    ##################
    elif (df['crop'] =='wint_wheat')  :
#        return np.mean(86,df['tmin_F'])-50

        return (((df['tmax_F']+df['tmin_F'])/2)-32)

    else: 
        return '0'   

#######################################################################

        
df['GDDf'] = df.apply(flag_df, axis = 1)

df["GDD_absf"]=(df['GDDf'].astype(float)).abs()

#wheat GDD condition if GDD>392 then high temp becomes 95

#orig=df

# again update absolute GDD column 

df.iat[13880,33] = 809  # potatoe because it was already planted
df.iat[31330,33] = 0   # winter wheat because it was planted in oct

df.iat[31422,33]=927.2

df1=pd.concat((df["ID"],df["year"],df["crop"],df["GDD_absf"]),axis=1)

df1['CumGDDf'] = df1.groupby(by=["ID","year","crop"])["GDD_absf"].cumsum()

df1 = df1.drop('GDD_absf', axis=1)

df["CumGDDf"]=df1["CumGDDf"]

new=df

df=pd.read_csv("crop_coeffi.csv")

df["cumGDD"]=new["CumGDDf"]



df.to_csv('pythoncumGDD.csv', index=False, header=True)

test=df

pivot=test.groupby(by=["week","crop"])["cumGDD"].mean()


#######################################################################
# add crop coefficient 

#######################################################################

os.chdir(r"G:\ML_ET\random_forest\input_fluxdata")
# excel sheets take long. Try 
#df=pd.read_excel('compile.xlsx', sheetname='all_data')
df=pd.read_csv("pythoncumGDD.csv")
orig=df

df.crop.unique()
def flag_df(df):
    
    ## corn 
    if (df['crop'] =='corn') and (df['cumGDD'] >= 379 and df['cumGDD']<500):
        return '0.1'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 500 and df['cumGDD']<639):
        return '0.18'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 639 and df['cumGDD']<788):
        return '0.35'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 788 and df['cumGDD']<939):
        return '0.51'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 939 and df['cumGDD']<1094):
        return '0.69'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 1094 and df['cumGDD']<1258):
        return '0.88'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 1258 and df['cumGDD']<1899):
        return '1.01'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 1899 and df['cumGDD']<2421):
        return '1.1'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 2421 and df['cumGDD']<2578):
        return '1.1'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 2578 and df['cumGDD']<2664) and df['week']<=44:
        return '1.1'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 2664 and df['cumGDD']<2687)and df['week']<=44:
        return '1.1'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 2687 and df['cumGDD']<2696)and df['week']<=44:
        return '1.1'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 2696 and df['cumGDD']<2701)and df['week']<=44:
        return '0.98'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 2701 and df['cumGDD']<2704)and df['week']<=44:
        return '0.60'
    elif (df['crop'] =='corn') and (df['cumGDD'] >= 2704)and df['week']<=44:
        return '.1'   
#soybean    
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 180 and df['cumGDD']<281):
        return '0.1'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 281 and df['cumGDD']<392):
        return '0.2'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 392 and df['cumGDD']<519):
        return '0.4'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 519 and df['cumGDD']<953):
        return '0.6'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 953 and df['cumGDD']<1108):
        return '0.9'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 1108 and df['cumGDD']<1270):
        return '1.0'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 1270 and df['cumGDD']<1604):
        return '1.10'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 1604 and df['cumGDD']<1769):
        return '1.10'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 1769 and df['cumGDD']<2059):
        return '1.10'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 2046 and df['cumGDD']<2431) and df['week']<=44:
        return '1.10'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 2415 and df['cumGDD']<2511)and df['week']<=44:
        return '0.9'
    elif (df['crop'] =='soy') and (df['cumGDD'] >= 2511 and df['cumGDD']<2512)and df['week']<=44:
        return '0.9'
#wheat    
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 180 and df['cumGDD']<252):
        return '0.1'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 252 and df['cumGDD']<395):
        return '0.5'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 395 and df['cumGDD']<538):
        return '0.9'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 538 and df['cumGDD']<967):
        return '1.03'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 967 and df['cumGDD']<1539):
        return '1.10'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 1539 and df['cumGDD']<1682):
        return '1.10'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 1682 and df['cumGDD']<1768):
        return '1.10'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 1768 and df['cumGDD']<2613)and df['week']<=44:
        return '1.10'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 2613 and df['cumGDD']<2832)and df['week']<=44:
        return '1.00'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 2832 and df['cumGDD']<2850) and df['week']<=44:
        return '0.50'
    elif (df['crop'] =='wheat') and (df['cumGDD'] >= 2850 and df['cumGDD']<3029)and df['week']<=44:
        return '0.1'
# winter wheat
    elif (df['crop'] =='wint_wheat') and (df['cumGDD'] >= 69 and df['cumGDD']<400):
        return '0'
    elif (df['crop'] =='wint_wheat') and (df['cumGDD'] >= 400 and df['cumGDD']<685):
        return '0.1'
    elif (df['crop'] =='wint_wheat') and (df['cumGDD'] >= 685 and df['cumGDD']<875):
        return '1.03'
    elif (df['crop'] =='wint_wheat') and (df['cumGDD'] >= 875 and df['cumGDD']<1075):
        return '1.1'
    elif (df['crop'] =='wint_wheat') and (df['cumGDD'] >= 1075 and df['cumGDD']<1575):
        return '1.1'
    elif (df['crop'] =='wint_wheat') and (df['cumGDD'] >= 1575 and df['cumGDD']<1825):
        return '0.5'
    elif (df['crop'] =='wint_wheat') and (df['cumGDD'] >1857):
        return '0.1'
# potato 1
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 34 and df['cumGDD']<180):
        return '0.1'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 180 and df['cumGDD']<472):
        return '0.175'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 472 and df['cumGDD']<809):
        return '0.45'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 809 and df['cumGDD']<924):
        return '0.65'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 924 and df['cumGDD']<1111):
        return '0.9'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 1111 and df['cumGDD']<1484):
        return '0.98'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 1484 and df['cumGDD']<2185):
        return '0.88'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 2185 and df['cumGDD']<2341)and df['week']<=44:
        return '0.86'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 2341 and df['cumGDD']<2776)and df['week']<=44:
        return '0.63'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 2776 and df['cumGDD']<2901) and df['week']<=44:
        return '0.42'
    elif (df['crop'] =='potat1') and (df['cumGDD'] >= 2901)and df['week']<=44:
        return '0.25'
##potato2    
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 34 and df['cumGDD']<79):
        return '0.1'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 79 and df['cumGDD']<272):
        return '0.175'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 272 and df['cumGDD']<665):
        return '0.45'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 665 and df['cumGDD']<852):
        return '0.65'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 852 and df['cumGDD']<982):
        return '0.9'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 982 and df['cumGDD']<1484):
        return '0.98'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 1484 and df['cumGDD']<2185):
        return '0.88'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 2185 and df['cumGDD']<2341)and df['week']<=44:
        return '0.86'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 2341 and df['cumGDD']<2776)and df['week']<=44:
        return '0.63'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 2901 and df['cumGDD']<2901) and df['week']<=44:
        return '0.42'
    elif (df['crop'] =='potat2') and (df['cumGDD'] >= 2901)and df['week']<=44:
        return '0.25'    
#clover kura
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 237 and df['cumGDD']<343):
        return '0.4'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 343 and df['cumGDD']<463):
        return '0.4'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 463 and df['cumGDD']<607):
        return '0.4'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 607 and df['cumGDD']<744):
        return '0.4'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 744 and df['cumGDD']<884):
        return '0.5'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 884 and df['cumGDD']<1041):
        return '0.85'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 1041 and df['cumGDD']<1697):
        return '0.9'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 1697 and df['cumGDD']<2209):
        return '1.0'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 2209 and df['cumGDD']<2359)and df['week']<=44:
        return '1.0'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 2359 and df['cumGDD']<2425)and df['week']<=44:
        return '1.0'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 2452 and df['cumGDD']<2457)and df['week']<=44:
        return '0.85'
    elif (df['crop'] =='kura') and (df['cumGDD'] >= 2457 and df['cumGDD']<2463)and df['week']<=44:
        return '0.1'
        
    else: 
        return '0'
 
    
df = df.drop('Flag', axis=1)
    
df['crop_coeff'] = df.apply(flag_df, axis = 1)

df.to_csv('crop_coeff.csv', index=False, header=True)

########################################################################
# add info about water use by different crops
#########################################################################

os.chdir(r"G:\ML_ET\random_forest\input_fluxdata")

df=pd.read_csv("crop_coeff.csv")

df.columns.tolist() 

df.crop.unique()

def wateruse(df):
    
    ## corn 
    if (df['crop'] =='wint_wheat'):
        return '1'
    elif (df['crop'] =='wheat'):
        return '2'
    elif (df['crop'] =='soy'):
        return '3'
    elif (df['crop'] =='kura'):
        return '4'
    elif (df['crop'] =='corn'):
        return '5'   
    elif (df['crop'] =='potat2'):
        return '6'
    elif (df['crop'] =='potat1'):
        return '7'
    else: 
        return '0'

 df['water_use'] = df.apply(wateruse, axis = 1)
  

########################################################################
# add info about crop managment till no till etc 
#########################################################################


df.ID.unique()

def wateruse(df):
    
    ## corn 
    if (df['ID'] =='wint_wheat'):
        return '1'
    elif (df['crop'] =='wheat'):
        return '2'
    elif (df['crop'] =='soy'):
        return '3'
    elif (df['crop'] =='kura'):
        return '4'
    elif (df['crop'] =='corn'):
        return '5'   
    elif (df['crop'] =='potat2'):
        return '6'
    elif (df['crop'] =='potat1'):
        return '7'
    else: 
        return '0'
####################################################################
##cover crop
#'MN_Ro2', 'MN_Ro3','MN_Ro6'         ####2
####################################################################
## tillage 
# 'IL_1B1', 'IA_Br1', 'IA_Br3' OH_CRT  'MI_T4' 'WI_CS' 'NB_Ne1  from 2006    ####1
########################################################################
# no tillage no cover
# 'MN_Ro1',   'MN_Ro5','IL_B02'     'MI_irrig', 'NB_Ne2', 'NB_Ne3' 'MI_nonirri' 'MI_T3'
 # 'IL_Bo1' ,'NB_Ne1 before 2006  
########################################################################

def crop_manag(df):    
    ## corn 
    if (df['ID']=='MN_Ro2' or df['ID']=='MN_Ro3' or df['ID']=='MN_Ro6'):
        return '2'
    
    elif (df['ID'] =='IL_1B1' or df['ID']=='IA_Br1' or df['ID'] =='OH_CRT' or df['ID']=='IA_Br3' or df['ID'] =='MI_T4'or df['ID']=='WI_CS' or (df['ID']=='NB_Ne1' and df['year']>=2006)):
        return '1'

    else:          
        return '0'
        
df['crop_cover'] = df.apply(crop_manag, axis = 1)
    
df.to_csv('crop_cover.csv', index=False, header=True)
#########################################################################
##planting dates

df.columns.tolist() 


def plant_week(df):
    
    ## corn 
    if (df['crop'] =='wint_wheat'):
        return '10'
    elif (df['crop'] =='wheat'):
        return '5.6'
    elif (df['crop'] =='soy'):
        return '5.6'
    elif (df['crop'] =='kura'):
        return '12'
    elif (df['crop'] =='corn'):
        return '5.4'   
    elif (df['crop'] =='potat2'):
        return '5.2'
    elif (df['crop'] =='potat1'):
        return '5.2'
    else: 
        return '0'

df['plantingweek'] = df.apply(plant_week, axis = 1)


def harvst_week(df):

    if (df['crop'] =='wint_wheat'):
        return '8.6'
    elif (df['crop'] =='wheat'):
        return '8.6'
    elif (df['crop'] =='soy'):
        return '10.6'
    elif (df['crop'] =='kura'):
        return '12'
    elif (df['crop'] =='corn'):
        return '10.8'   
    elif (df['crop'] =='potat2'):
        return '9.4'
    elif (df['crop'] =='potat1'):
        return '9.4'
    else: 
        return '0'

df['harvst_week'] = df.apply(harvst_week, axis = 1)


df.to_csv('complte_data.csv', index=False, header=True)

#########################################################################

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from math import*

os.chdir(r"G:\ML_ET\random_forest\input_fluxdata")
df=pd.read_csv("complte_data.csv")

#########################################################################################################

### code to include soil type to IDs


import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from math import*




os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")

df = pd.read_excel('results.xlsx', sheetname='pre_test')

df = pd.read_excel('results.xlsx', sheetname='pre_train')









