# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html
#heat map
#https://jovianlin.io/data-visualization-seaborn-part-2/

#https://github.com/turo/public-resources/blob/master/blog-posts/how-not-to-use-random-forest/How-not-to-use-random-forest.ipynb


# use 3.5 for machine learning
#https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#from rfpimp import *
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from math import sqrt

from sklearn.model_selection import train_test_split
#RF_Model = RandomForestRegressor(n_estimators=100,
os.chdir(r"G:\ML_ET\random_forest\input_fluxdata")
df=pd.read_excel('compile.xlsx', sheetname='adjust_forecast')

dat=df
cols = dat.columns.tolist()
#select till longwave
df=dat.iloc[:,3:31]

df['sur_refl_b01_1']=df['sur_refl_b01_1']*0.0001
df['sur_refl_b02_1']=df['sur_refl_b02_1']*0.0001
df['sur_refl_b03_1']=df['sur_refl_b03_1']*0.0001

df['sur_refl_b01_1']=np.where(df["sur_refl_b01_1"]<-100,np.nan, df["sur_refl_b01_1"])
df['sur_refl_b02_1']=np.where(df["sur_refl_b02_1"]<-100,np.nan, df["sur_refl_b02_1"])
df['sur_refl_b03_1']=np.where(df["sur_refl_b03_1"]<-100,np.nan, df["sur_refl_b03_1"])

df['sur_refl_b01_1']=np.where(df["sur_refl_b01_1"]>16000,np.nan, df["sur_refl_b01_1"])
df['sur_refl_b02_1']=np.where(df["sur_refl_b02_1"]>16000,np.nan, df["sur_refl_b02_1"])
df['sur_refl_b03_1']=np.where(df["sur_refl_b03_1"]>16000,np.nan, df["sur_refl_b03_1"])

df['ALBEDO_1']=np.where(df["ALBEDO_1"]<0,np.nan, df["ALBEDO_1"])
df.isnull().values.any()

df2=df
df2=df.fillna(df.rolling(2,1).mean())
df2.isnull().values.any()
df=df2




df["NDV1"]=(df['sur_refl_b02_1']-df['sur_refl_b01_1'])/(df['sur_refl_b02_1']+df['sur_refl_b01_1'])
df["NDV1"]=np.where(df["NDV1"]<-1,np.nan, df["NDV1"])
df["NDV1"]=np.where(df["NDV1"]>1,np.nan, df["NDV1"])
df=df.fillna(df.rolling(2,1).mean())
df=df.fillna(df.rolling(4,1).mean())
df.isnull().values.any()

df["EVI"]=2.5*(df['sur_refl_b02_1']-df['sur_refl_b01_1'])/(df['sur_refl_b02_1']+6*df['sur_refl_b01_1']-7.5*df['sur_refl_b03_1']+1)
df["EVI"]=np.where(df["EVI"]<-1,np.nan, df["EVI"])
df["EVI"]=np.where(df["EVI"]>1,np.nan, df["EVI"])
df=df.fillna(df.rolling(2,1).mean())
df=df.fillna(df.rolling(4,1).mean())
df.isnull().values.any()

df["SAVI"]=((df['sur_refl_b02_1']-df['sur_refl_b01_1'])/(df['sur_refl_b02_1']+df['sur_refl_b01_1']+0.5))*(1+0.5)
df["SAVI"]=np.where(df["SAVI"]<-1,np.nan, df["SAVI"])
df["SAVI"]=np.where(df["SAVI"]>1,np.nan, df["SAVI"])
df=df.fillna(df.rolling(2,1).mean())
df=df.fillna(df.rolling(4,1).mean())
df.isnull().values.any()

cols = df.columns.tolist()

#plt.scatter(output.iloc[:,0],output.iloc[:,2],color='red')
#plt.scatter(output.iloc[:,0],output.iloc[:,3],color='blue')

X=pd.concat((df["soil_type"],df["SolarZenith_1"],df["prcp_60"],df["tmax_(deg c)"],df["tmin_(deg c)"],df["vp_(Pa)"],df["wind_speed"],df["SW_Wm-2"],df["NDV1"],df["EVI"],df["SAVI"],df["min_temp_For"],df["max_temp_for"],df["SWforeast"],df["forcas_prec60"]),axis=1)

X=pd.concat((df["soil_type"],df["SolarZenith_1"],df["prcp_60"],df["tmax_(deg c)"],df["tmin_(deg c)"],df["vp_(Pa)"],df["wind_speed"],df["SW_Wm-2"],df["NDV1"],df["EVI"],df["SAVI"],df["min_temp_For"],df["max_temp_for"],df["SWforeast"],df["forcas_prec60"]),axis=1)

Y=pd.DataFrame(df["ET_obs"])

Y=Y.iloc[:,0]


reframed=pd.concat((Y,X),axis=1)
cols = reframed.columns.tolist()

reframed.reset_index(level=0, inplace=True)
input_da=reframed

dataset = input_da.iloc[:,1:23]

cols = dataset.columns.tolist()

values = dataset.values

#############################################################################
#autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig=plt.figure(figsize=(8,5))
ax1=fig.add_subplot(211)
#fig=plot_acf(y,lags=200,ax=ax1)
#ax2=fig.add_subplot(212)
#fig=plo_pacf(y,lags=30,ax=ax2)
###############################################################################


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
# integer encode direction
#encoder = LabelEncoder()  # only need for catagorical data
#values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)



reframed = series_to_supervised(scaled,1, 7)
# no lags
#reframed = series_to_supervised(scaled,1, 1)

# # of* scaled shape*(time step)

#reframed.drop(reframed.columns[22:660], axis=1, inplace=True)
#reframed.drop(reframed.columns[23:681], axis=1, inplace=True)



#reframed.drop(reframed.columns[22:660], axis=1, inplace=True)
#reframed.drop(reframed.columns[23:681], axis=1, inplace=True)


#########################################################################

# one week forcast 
#x/(prediction time)=# number of predictors 

# frame as supervised learning

# # of* scaled shape*(time step)



#reframed = series_to_supervised(scaled, 1, 10)
#reframed.drop(reframed.columns[17:112], axis=1, inplace=True)

reframed = series_to_supervised(scaled, 1, 1)
#reframed.drop(reframed.columns[17:40], axis=1, inplace=True)
#reframed.drop(reframed.columns[17:112], axis=1, inplace=True)
reframed.drop(reframed.columns[17:32], axis=1, inplace=True)

#reframed.drop(reframed.columns[0], axis=1, inplace=True)
#reframed.drop(reframed.columns[16:32], axis=1, inplace=True)

#reframed.drop(reframed.columns[17:110], axis=1, inplace=True)

#reframed.drop(reframed.columns[17:30], axis=1, inplace=True)
                          # same as column used in previous line: 
#reframed = series_to_supervised(scaled, 1, 9)
#reframed.drop(reframed.columns[51:112], axis=1, inplace=True)
#reframed.drop(reframed.columns[51:108], axis=1, inplace=True)

#                        caled.shape[1]:((scaled.shape[1]-1)* time step


#reframed.drop(reframed.columns[16:1], axis=1, inplace=True)

#reframed.drop(reframed.columns[23:150], axis=1, inplace=True)

#######################################################################
values = reframed.values
n_train_hours = 2360
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]

# train x does not include last column of reframed because last column of reframes is train y
#(446, 22)   (446,)
test_X, test_y = test[:, :-1], test[:, -1]
#(191, 21)   (191,)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#(446, 1, 21)

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#(191, 1, 21)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#(446, 1, 21) (446,) (191, 1, 21) (191,)



n_steps=1
n_features=17
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=1))  # pool_size=(kernal size)
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# make a prediction
yhat = model.predict(test_X)
#(191,1)
#(35039, 1)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#191,21
#(35039, 8)
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)

#(191, 21)
#(35039, 8)
inv_yhat = scaler.inverse_transform(inv_yhat)
#(35039, 8)
inv_yhat = inv_yhat[:,0]
#(35039,)
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
#(35039, 1)
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#(35039, 8)
inv_y = scaler.inverse_transform(inv_y)
#(35039, 8)
inv_y = inv_y[:,0]
#(35039,)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)



import matplotlib.pyplot as plt
plt.plot(inv_yhat,color='green', label="prediction") 
plt.plot(inv_y,color='red',label="observed")
plt.ylabel('GW depth below growth (ft)',size=10)
plt.xlabel('no of observation',size=10)
plt.legend()
plt.legend(loc=1, fontsize = 'small')
plt.title('Recurrent Neural Network',size=12)
















import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
from math import sqrt
from numpy import split
from numpy import array
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
#from keras.layers.merge import concatenate

from numpy import concatenate
ref=pd.concat((X,Y),axis=1)
#values = ref.astype('float32')
values=ref


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg




values =ref.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)



reframed = series_to_supervised(scaled, 1, 7)
reframed.drop(reframed.columns[17:135], axis=1, inplace=True)

# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
scaled = scaler.transform(values)
#inversed = scaler.inverse_transform(normalized)




#scaler = StandardScaler()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(values)
# normalize the dataset and print
scaled = scaler.transform(values)

#scaled=values

# frame as supervised learning
# bascially no lagging 
#reframed = series_to_supervised(scaled, 1, 0)
# drop columns we don't want to predict
#print(reframed.head())
# try without lagging first time


###############################################################################

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#sav=scaled
#reframed = scaled
### no lags
reframed = series_to_supervised(scaled,1, 0)

values = reframed.values

n_train_hours = 1825  #1965
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

n_steps=1

n_steps = 1  # no lags or previous timestep

#train_X, train_y = split_sequences(train,n_steps)
#test_X, test_y = split_sequences(test,n_steps)
train_X, train_y=train[:, :-1], train[:, -1]
test_X, test_y=test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#(12053, 1, 6)
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

#train_X, train_y = split_sequences(train,n_steps)
#test_X, test_y = split_sequences(test,n_steps)

n_features = train_X.shape[2]  # number of input variables

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#(1825, 1, 16) (1825,) (915, 1, 16) (915,)

#(915, 1, 16)
#(1825, 1, 16)  check test x and train x dimensions should match
# define model
#kernal size is equal to n_steps
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=1))  # pool_size=(kernal size)
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
                                  #1              # 6
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# make a prediction
# fit model
# demonstrate prediction
yhat = model.predict(test_X)

#
#print(yhat)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
  #915,16          #915,1,16              
inv_yhat = concatenate((yhat,test_X[:,0: n_features]), axis=1)
  #  915,17                     #915,1       915,16
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
#12053 ,1 
inv_y = concatenate((test_y, test_X[:,0: n_features]), axis=1)
                    #12053 ,1         12053,16
#inv_y = scaler.inverse_transform(inv_y)
inv_y= scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

############################################################################
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Train RMSE: %.3f' % rmse)
plt.plot(inv_yhat,color='green', label="prediction") 
plt.plot(inv_y,color='red',label="observed")
plt.ylabel('GW depth below growth (ft)',size=10)
plt.xlabel('no of observation',size=10)
plt.legend()
plt.legend(loc=1, fontsize = 'small')
plt.title('Recurrent Neural Network',size=12)
# define input sequence
r2_score(inv_y, inv_yhat)

######################################################################

#training data
# invert scaling for actual
yhat = model.predict(train_X)

train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
  #915,16          #915,1,16              
inv_yhat = concatenate((yhat,train_X[:,0: n_features]), axis=1)
  #  915,17                     #915,1       915,16
inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

train_y = train_y.reshape((len(train_y), 1))
#12053 ,1 
inv_y = concatenate((train_y, train_X[:,0: n_features]), axis=1)
                    #12053 ,1         12053,16
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


###########12053

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores



# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)


# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 1, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model



# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat



# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores



