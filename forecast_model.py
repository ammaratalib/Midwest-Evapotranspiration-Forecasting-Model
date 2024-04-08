# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:22:55 2019

@author: Ammara
"""

import os

import numpy as np

import pandas as pd

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

#RF_Model = RandomForestRegressor(n_estimators=100,
 #                                max_features="auto", oob_score=True)
#from hydroeval import *


def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    imp = np.array(imp)
    I = pd.DataFrame(
        data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I


def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1
    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score



os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata")

#df.to_csv('summer_forecast.csv', index=False, header=True)

df=pd.read_csv("summer_forecast.csv")

#os.chdir(r"C:\ammara_MD\ML_ET\random_forest\input_fluxdata")


#df=pd.read_csv("summer_data.csv")

#data=pd.concat((df,dff),axis=1)
data=df
backup=data

df=data

id_un=['IL_1B1', 'IL_B02', 'NB_Ne2', 'NB_Ne3', 'WI_CS', 'IA_Br1', 'MN_Ro1',
       'MN_Ro2', 'MN_Ro3', 'MN_Ro5', 'MI_T4', 'MI_nonirri', 'OH_CRT', 'MI_T3',
       'MI_irrig', 'MN_Ro6', 'NB_Ne1', 'IA_Br3', 'IL_Bo1']


df['ET_shifted'] = df.groupby(['ID'])['ET'].shift(3)

df = df[pd.notnull(df['ET_shifted'])]

back=df

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
back=df
#plt.scatter(output.iloc[:,0],output.iloc[:,2],color='red')
#plt.scatter(output.iloc[:,0],output.iloc[:,3],color='blue')

# all variable
### remove minium temp
######################################################################

### met_var
met=pd.concat((df['prcp_7'],df['prcp_15'],df['prcp_30'],df['vp'],df['tmax_F'],\
             df['LW'],df['SW'],df['wind'],df['SolarZenith_1'],df['ALBEDO_1'],\
             df['soil'],df['irr_nonirr'],df['crop_cover'],df['crop_coeff'],df['cumGDD'],
             df['EVI']
              ),axis=1)

forcas1=pd.concat((df['LD1pcpcntr'],df['LD2pcpcntr'],df['LD3pcpcntr'],df['LD1swcntr'],df['LD2swcntr'],\
                   df['LD3swcntr'],df['LD1tmincntr1'],df['LD2tmincntr'],df['LD3tmincntr'],df['LD1tmaxcntr1'],\
                df['LD2tmaxcntr'],df['LD3tmaxcntr']),axis=1)


forcas2=pd.concat((df['LD1pcp1'],df['LD2pcp1'],df['LD3pcp1'],df['LD1sw1'],df['LD2sw1'],\
                   df['LD3sw1'],df['LD1tmin1'],df['LD2tmin1'],df['LD3tmin1'],df['LD1tmax1'],\
                   df['LD2tmax1'],df['LD3tmax1']),axis=1)
                   

forcas3=pd.concat((df['LD1pcp2'],df['LD2pcp2'],df['LD3pcp2'],df['LD1sw2'],df['LD2sw2'],\
                   df['LD3sw2'],df['LD1tmin2'],df['LD2tmin2'],df['LD3tmin2'],df['LD1tmax2'],\
                   df['LD2tmax2'],df['LD3tmax2']),axis=1)

forcas4=pd.concat((df['LD1pcp3'],df['LD2pcp3'],df['LD3pcp3'],df['LD1sw3'],df['LD2sw3'],\
                   df['LD3sw3'],df['LD1tmin3'],df['LD2tmin3'],df['LD3tmin3'],df['LD1tmax3'],\
                   df['LD2tmax3'],df['LD3tmax3']),axis=1)

forcas5=pd.concat((df['LD1pcp4'],df['LD2pcp4'],df['LD3pcp4'],df['LD1sw4'],df['LD2sw4'],\
                   df['LD3sw4'],df['LD1tmin4'],df['LD2tmin4'],df['LD3tmin4'],df['LD1tmax4'],\
                   df['LD2tmax4'],df['LD3tmax4']),axis=1)


forcas6=pd.concat((df['LD1pcp5'],df['LD2pcp5'],df['LD3pcp5'],df['LD1sw5'],df['LD2sw5'],\
                   df['LD3sw5'],df['LD1tmin5'],df['LD2tmin5'],df['LD3tmin5'],df['LD1tmax5'],\
                   df['LD2tmax5'],df['LD3tmax5']),axis=1)


forcas7=pd.concat((df['LD1pcp6'],df['LD2pcp6'],df['LD3pcp6'],df['LD1sw6'],df['LD2sw6'],\
                   df['LD3sw6'],df['LD1tmin6'],df['LD2tmin6'],df['LD3tmin6'],df['LD1tmax6'],\
                   df['LD2tmax6'],df['LD3tmax6']),axis=1)


forcas8=pd.concat((df['LD1pcp7'],df['LD2pcp7'],df['LD3pcp7'],df['LD1sw7'],df['LD2sw7'],\
                   df['LD3sw7'],df['LD1tmin7'],df['LD2tmin7'],df['LD3tmin7'],df['LD1tmax7'],\
                   df['LD2tmax7'],df['LD3tmax7']),axis=1)


forcas9=pd.concat((df['LD1pcp8'],df['LD2pcp8'],df['LD3pcp8'],df['LD1sw8'],df['LD2sw8'],\
                   df['LD3sw8'],df['LD1tmin8'],df['LD2tmin8'],df['LD3tmin8'],df['LD1tmax8'],\
                   df['LD2tmax8'],df['LD3tmax8']),axis=1)

forcas10=pd.concat((df['LD1pcp9'],df['LD2pcp9'],df['LD3pcp9'],df['LD1sw9'],df['LD2sw9'],\
                   df['LD3sw9'],df['LD1tmin9'],df['LD2tmin9'],df['LD3tmin9'],df['LD1tmax9'],\
                   df['LD2tmax9'],df['LD3tmax9']),axis=1)


forcas11=pd.concat((df['LD1pcp10'],df['LD2pcp10'],df['LD3pcp10'],df['LD1sw10'],df['LD2sw10'],\
                   df['LD3sw10'],df['LD1tmin10'],df['LD2tmin10'],df['LD3tmin10'],df['LD1tmax10'],\
                   df['LD2tmax10'],df['LD3tmax10']),axis=1)

###########################################################################################################

###  16 variables prediction model 26331  training is so 70% training  70.187  till MI 3 2016 ends

##### 28 variables forecasting   26274  training is  70.1796  till MI 3 2016 ends

plt.plot(df["ET"],'r')
plt.plot(df["ET_shifted"],'b')


X=pd.concat((met,forcas6),axis=1)

Y=pd.DataFrame(df["ET_shifted"])

Y=Y.iloc[:,0]

#X_train = X.loc[df.index <= 18439]  # one row less than row ID

X_train = X[X.index <= 18439].copy()



# just to make sure you are correct
X_train.tail(10)

 
#y_train = Y.loc[df.index <= 18439]              

y_train = Y[Y.index <= 18439].copy()            

#y_train=y_train["ET"]

#X_test = X.loc[df.index > 18439]    
#y_test = Y.loc[df.index > 18439]   

X_test = X[X.index > 18439].copy()
y_test = Y[Y.index > 18439].copy()             
           

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# so far best
# model with more parameters
#rf = RandomForestRegressor(criterion='mse',n_estimators=300,min_samples_leaf=10,min_samples_split=8,n_jobs=-1,oob_score=True,max_depth=17,max_features='auto',bootstrap=True)

# model with less parameters

rf = RandomForestRegressor(criterion='mse',n_estimators=300,min_samples_leaf=7,min_samples_split=8,n_jobs=-1,oob_score=True,max_depth=17,max_features='auto',bootstrap=True)

rf.fit(X_train, y_train)

# pick between either default or oermuted score
# default score
#names =X_train.columns.tolist()
#print ("Features sorted by their score:")
#print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
 #            reverse=True))
#k=sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
 #            reverse=True)
# permuted score
#imp=(permutation_importances(rf, X_train, y_train,oob_regression_r2_score))
###############################################################################
labels = y_train#[:, None]
features = X_train[:]
#rf=RF_Model.fit(features, labels) 

#########################################################################

X_test_predict=pd.DataFrame(
    rf.predict(X_test[:])).rename(
    columns={0:'predicted_ET'}).set_index('predicted_ET')

X_train_predict=pd.DataFrame(
    rf.predict(X_train[:])).rename(
    columns={0:'predicted_ET'}).set_index('predicted_ET')

RF_predict = X_train_predict.append(X_test_predict)
RF_predict .reset_index(level=0, inplace=True)


X_test_predict.reset_index(level=0, inplace=True)
r2_score(y_test,X_test_predict)

X_train_predict.reset_index(level=0, inplace=True)
r2_score(y_train,X_train_predict)


y_train_predicted = rf.predict(X_train)
y_test_predicted_pruned_trees = rf.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_predicted)
mse_test = mean_squared_error(y_test, y_test_predicted_pruned_trees)
print("RF with pruned trees, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))

print("RF with pruned trees, Train RMSE: {} Test RMSE: {}".format(sqrt(mse_train), sqrt(mse_test)))
