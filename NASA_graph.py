# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:54:26 2018

@author: ammara
"""





########Graph2

import os 

os.chdir(r"G:\ML_ET\Minnesota")


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  

#####graph 1
import matplotlib.pyplot as plt
import numpy as np


####################################################################
#graph1         long term data
#################################################################
import os 
os.chdir(r"C:\wisconsin\machine_learning\machine_learningWCS\input_data\hancock")


df=pd.read_excel('AGU.xlsx', sheetname='brier_test')

def NS(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
   # s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)

df=pd.read_excel('AGU.xlsx', sheetname='brier_train')


NS(df["pred_all"],df["obs"])

NS(df["pred_soilmois"],df["obs"])

NS(df["pred_soil_temp"],df["obs"])
NS(df["pred_RH"],df["obs"])

NS(df["pred_Rn"],df["obs"])

NS(df["pred_temp"],df["obs"])

NS(df["pred_prec"],df["obs"])

NS(df["pred_pdo"],df["obs"])




#import numpy as np
from sklearn.metrics import brier_score_loss

from sklearn.metrics import brier_score_loss

# plot impact of logloss with imbalanced datasets
from sklearn.metrics import log_loss
probs = model.predict_proba(df["obs"])

probs=[log_loss(df["obs"], [y for x in range(len(df["obs"]))]) for y in df["pred_all"]]


probs=[log_loss(df["obs"], [y for x in range(len(df["obs"]))]) for y in df["pred_all"]]


losses = [brier_score_loss([1], [x], pos_label=[1]) for x in df["pred_all"]]






# define an imbalanced dataset










##########################################################################


os.chdir(r"G:\ML_ET")

df=pd.read_excel('nasa_minn.xlsx', sheetname='30day_min_tra')

plt.rcParams['figure.figsize'] = (3.5, 3)
fig, ax = plt.subplots()
ax.plot(df["TIMESTAMP"], df["obs_train"], color='green',label="Observed",linestyle=":")
ax.plot(df["TIMESTAMP"], df["pred_train"],  color='blue',label="Predicted RMSE (0.37)",linestyle="--" )
ax.axvline(pd.to_datetime('04-22-2018'), color='red', linestyle='--', lw=2,label="Train-Test Split")
ax.legend(loc=2, fontsize = 'small')
ax.set_ylabel('Evaaranspiration (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Daily Timestep')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(9)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 9)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.set_ylim(0,0.3)
ax.set_title("30-Day ET Forecast in Corn/Soybean Field",fontsize=9)
ax = plt.gca()
#ax.invert_yaxis()
fig.autofmt_xdate() 
plt.show()
os.chdir(r"G:\NASA\NASA_proposal")

fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()






######################################################################

### 7 day forecast
#########################################################################
os.chdir(r"G:\ML_ET")

df=pd.read_excel('nasa_minn.xlsx', sheetname='7day_min_trai')

plt.rcParams['figure.figsize'] = (3.5, 3)
fig, ax = plt.subplots()
ax.plot(df["TIMESTAMP"], df["obs_train"], color='green',label="Observed",linestyle=":")
ax.plot(df["TIMESTAMP"], df["pred_train"],  color='blue',label="Predicted RMSE (0.36)",linestyle="--" )
ax.axvline(pd.to_datetime('03-30-2018'), color='red', linestyle='--', lw=2,label="Train-Test Split")

ax.legend(loc=2, fontsize = 'small')
#ax.legend( ('Observed', 'Predicted', 'Train-Test Split','R2 ='))
ax.set_ylabel('Evapotranspiration (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Daily Timestep')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(9)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 9)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)

ax.set_ylim(0,0.3)

ax.set_title("7 Day ET Forecast in Corn/Soybean Field",fontsize=9)
ax = plt.gca()
#ax.invert_yaxis()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
os.chdir(r"G:\NASA\NASA_proposal")

fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

##################################################################
##potato field

###################################################################
os.chdir(r"G:\ML_ET")

df=pd.read_excel('nasa_pot.xlsx', sheetname='7day_pot_trai')

plt.rcParams['figure.figsize'] = (3.5, 3)
fig, ax = plt.subplots()
ax.plot(df["TIMESTAMP"], df["obs_train"], color='green',label="Observed",linestyle=":")
ax.plot(df["TIMESTAMP"], df["pred_train"],  color='blue',label="Predicted RMSE (0.36)",linestyle="--" )
ax.axvline(pd.to_datetime('12-01-2018'), color='red', linestyle='--', lw=2,label="Train-Test Split")

ax.legend(loc=2, fontsize = 'small')
#ax.legend( ('Observed', 'Predicted', 'Train-Test Split','R2 ='))
ax.set_ylabel('Evapotranspiration (inches)')  # we already handled the x-label with ax1
ax.set_xlabel('Daily Timestep')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(9)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 9)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)

ax.set_ylim(0,0.3)

ax.set_title("7 Day ET Forecast in Corn/Soybean Field",fontsize=9)
ax = plt.gca()
#ax.invert_yaxis()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
os.chdir(r"G:\NASA\NASA_proposal")

fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()







12/4/2018





######################################################################

#GRAPH 2  soil moisture data
########################################################################

df = pd.read_excel('AGU.xlsx', sheetname='soil_mois_tog')


plt.rcParams['figure.figsize'] = (7, 6)
fig, ax = plt.subplots()
ax.plot(df["train"], df["obs"], color='red',label="Observed",linestyle=":")
ax.plot(df["train"], df["pred_all"],  color='blue',label="Predicted",linestyle=":" )
ax.axvline(pd.to_datetime('2014-11-01'), color='green', linestyle='--', lw=2,label="Train-Test Split")
ax.legend(loc=1, fontsize = 'large')
ax.set_ylabel('GW depth below ground (ft)')  # we already handled the x-label with ax1

ax.set_xlabel('Daily Timestep')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 15)
ax.set_ylim(4,17)

ax.set_title("One Month Ground water Depth Forecast",fontsize=15)
ax = plt.gca()
ax.invert_yaxis()

#fig.autofmt_xdate() 
plt.show()


##################### GRAPH 3








###########################################################################

# residula plot not for AGU plot to filled in color 
#fig, ax = plt.subplots()
#ax.scatter(names, df["resi_relHu"],color='red', Label="RH",marker="p",s=abs(df["resi_relHu"])*200)
#ax.scatter(names, df["All predictors"],color='red', Label="All Predictors",marker="p",s=100,)
#ax.scatter(names, df["Soil Mois"],color='blue',Label="Soil Moisture",marker="+",s=100)
#ax.scatter(names, df["Soil Temp"],color='green',Label="Soil Temp",marker="*",s=100)
#############################################################################


#residual_test

os.chdir(r"C:\wisconsin\machine_learning\machine_learningWCS\input_data\hancock")

df = pd.read_excel('AGU.xlsx', sheetname='residual_test')
df.columns=['month','All predictors','Soil Mois','Soil Temp','RH','Net Rad',
              'Air Temp','Precip','PDO']

import matplotlib.pyplot as plt

names =['Winter',' Spring','Summer','Fall']
#markerfacecolor='none', markeredgecolor='red'

fig, ax = plt.subplots()
#ax.scatter(names, df["resi_relHu"],color='red', Label="RH",marker="p",s=abs(df["resi_relHu"])*200)
ax.scatter(names, df["All predictors"],facecolors='none',edgecolors='red', Label="All Predictors",marker="p",s=100,linewidth=2)
ax.scatter(names, df["Soil Mois"],facecolors='none',edgecolors='blue',Label="Soil Moisture",marker="8",s=100,linewidth=2)
ax.scatter(names, df["Soil Temp"],facecolors='none',edgecolors='brown',Label="Soil Temp",marker="*",s=100,linewidth=2)
ax.scatter(names, df["RH"],facecolors='none',edgecolors='purple',Label="Relative Humidity",marker="s",s=100,linewidth=2)
ax.scatter(names, df["Net Rad"],facecolors='none',edgecolors="magenta",Label="Net Radiations",marker="o",s=100,linewidth=2)
ax.scatter(names, df["Air Temp"],facecolors='none',edgecolors='black',Label="Air Temp",marker="h",s=100,linewidth=2)
ax.scatter(names, df["Precip"],facecolors='none',edgecolors="darkgreen",Label="Precipitation",marker="<",s=100,linewidth=2)
ax.scatter(names, df["PDO"],facecolors='none',edgecolors='grey',Label="PDO",marker="D",s=100,linewidth=2)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Residuals (ft)')  # we already handled the x-label with ax1
#ax.set_xlabel('Daily Timestep')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 15)
ax.set_title("Residuals for One month Forecast Model for Test Data",fontsize=15)
ax = plt.gca()
ax.set_ylim(0,-0.8)
ax.invert_yaxis()

ax.invert_yaxis()

#fig.autofmt_xdate() 
plt.show()









import numpy as np
from sklearn.metrics import brier_score_loss

from sklearn.metrics import brier_score_loss
...
model = ...
testX, testy = ...
# predict probabilities
probs = model.predict_proba(testX)
# keep the predictions for class 1 only
probs = probs[:, 1]
# calculate bier score
loss = brier_score_loss(testy, probs)









 y_true = np.array([0, 1, 1, 0])

brier_score_loss(y_true, y_prob)  

>>> brier_score_loss(y_true, 1-y_prob, pos_label=0)  
0.037...
>>> brier_score_loss(y_true_categorical, y_prob,                          pos_label="ham")  
0.037...
>>> brier_score_loss(y_true, np.array(y_prob) > 0.5)












os.chdir(r"C:\wisconsin\machine_learning\machine_learningWCS\input_data\hancock")

df = pd.read_excel('AGU.xlsx', sheetname='residual_train')
df.columns=['month','All predictors','Soil Mois','Soil Temp','RH','Net Rad',
              'Air Temp','Precip','PDO']




#ax.scatter(names, df["resi_relHu"],color='red', Label="RH",marker="p",s=abs(df["resi_relHu"])*200)
fig, ax = plt.subplots()
#ax.scatter(names, df["resi_relHu"],color='red', Label="RH",marker="p",s=abs(df["resi_relHu"])*200)
ax.scatter(names, df["All predictors"],facecolors='none',edgecolors='red', Label="All Predictors",marker="p",s=100,linewidth=2)
ax.scatter(names, df["Soil Mois"],facecolors='none',edgecolors='blue',Label="Soil Moisture",marker="8",s=100,linewidth=2)
ax.scatter(names, df["Soil Temp"],facecolors='none',edgecolors='brown',Label="Soil Temp",marker="*",s=100,linewidth=2)
ax.scatter(names, df["RH"],facecolors='none',edgecolors='purple',Label="Relative Humidity",marker="s",s=100,linewidth=2)
ax.scatter(names, df["Net Rad"],facecolors='none',edgecolors="magenta",Label="Net Radiations",marker="o",s=100,linewidth=2)
ax.scatter(names, df["Air Temp"],facecolors='none',edgecolors='black',Label="Air Temp",marker="h",s=100,linewidth=2)
ax.scatter(names, df["Precip"],facecolors='none',edgecolors="darkgreen",Label="Precipitation",marker="<",s=100,linewidth=2)
ax.scatter(names, df["PDO"],facecolors='none',edgecolors='grey',Label="PDO",marker="D",s=100,linewidth=2)

#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Residuals (ft)')  # we already handled the x-label with ax1
#ax.set_xlabel('Daily Timestep')  # we already handled the x-label with ax1
ax.yaxis.label.set_size(15)
ax.xaxis.label.set_size(15) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 15)
ax.set_title("Residuals for One month Forecast Model for Train Data",fontsize=15)
ax = plt.gca()
ax.set_ylim(0,-0.8)
ax.invert_yaxis()

ax.invert_yaxis()

#fig.autofmt_xdate() 
plt.show()







































