
#####################################################################################################################
import os

os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")


import os
os.environ['PROJ_LIB'] = r"C:\Users\Ammara\Anaconda3\pkgs\proj4-5.1.0-hfa6e2cd_1\Library\share"

pyproj_datadir = os.environ['PROJ_LIB']
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

from bokeh.plotting import figure, show
from bokeh.sampledata.us_states import data as states
from bokeh.models import ColumnDataSource, Range1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

### merge both together

   #    'avg_rmseforL1rfET',  'avg_rmseforL2rfET', 'avg_rmseforL3rfET',
   #    'avg_rmseforL1rnnET', 'avg_rmseforL2rnnET', , 'avg_rmseforL3rnnET'],
   #   dtype='object')
   
df=pd.read_csv('fore_test.csv')



def r2_rmse( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    rmse = np.sqrt( mean_squared_error( g['ET3_obs']*25.4, g['rnnL3ET3']*25.4 ) )
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return pd.Series( dict(rmse = rmse ) )

df.groupby( 'month' ).apply( r2_rmse ).reset_index()



df=pd.read_csv('fore_test.csv')





def r2( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    r2 = r2_score(g['ET3_obs'], g['rnnL3ET3'])
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return pd.Series( dict(r2 = r2 ) )

df.groupby( 'month' ).apply( r2).reset_index()



def pbias_( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    pb=(np.sum(pre-obs)/np.sum(obs))*100
   # r2 = r2_score(['ET3_obs']*25.4, g['rnnL3ET3'])
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return pd.Series( dict(pb = pb ) )

df.groupby( 'month' ).apply( pbias_).reset_index()






def anova( g):
#    r2 = r2_score( g['ET2_obs'], g['rfL3ET2'] )
    pval = scipy.stats.f_oneway(g.rfL3ET3,g.rnnL3ET3)
 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return pd.Series( dict(pval = pval ) )

df.groupby( 'month' ).apply( anova ).reset_index()

##########################################################################################################################
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")

df=pd.read_csv('forecast_results.csv')  #### forecast results by month


Index(['month', 'L1rf_day1', 'L1rnn_day1', 'L1rf_day2', 'L1rnn_day2',
       'L1rf_day3', 'L1rnn_day3', 'avg_model1_rf', 'avg_model1_rnn',
       'L2rf_day1', 'L2rnn_day1', 'L2rf_day2', 'L2rnn_day2', 'L2rf_day3',
       'L2rnn_day3', 'avg_model2_rf', 'avg_model2_rnn', 'L3rf_day1',
       'L3rnn_day1', 'L3rf_day2', 'L3rnn_day2', 'L3rf_day3', 'L3rnn_day3',
       'avg_model3_rf', 'avg_model3_rnn'],
      dtype='object')


plt.plot(df["L1rf_day1"])



############################################################################
# model 1
### lead time 1 day 1
##########################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L1rf_day1"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 1 ET")
ax.plot(df["L1rnn_day1"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 1 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.2)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("One day lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################################
## model 1
### lead time 1 day 2

####################################################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L1rf_day2"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 2 ET")
ax.plot(df["L1rnn_day2"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 2 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("One day lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################################
## model 1
### lead time 1 day 3

####################################################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L1rf_day3"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 3 ET")
ax.plot(df["L1rnn_day3"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 3 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("One day lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
##########################################################################################################################

############################################################################
# model 2
### lead time 2 day 1
##########################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L2rf_day1"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 1 ET")
ax.plot(df["L2rnn_day1"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 1 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.2)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("Two days lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################################
## model 2
### lead time 2 day 2

####################################################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L2rf_day2"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 2 ET")
ax.plot(df["L2rnn_day2"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 2 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("Two days lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################################
## model 2
### lead time 3 day 3

####################################################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L3rf_day3"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 3 ET")
ax.plot(df["L3rnn_day3"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 3 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("Two days lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
##########################################################################################################################

############################################################################
# model 3
### lead time 3 day 1
##########################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L3rf_day1"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 1 ET")
ax.plot(df["L3rnn_day1"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 1 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.2)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("Three days lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################################
## model 3
### lead time 3 day 2

####################################################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L3rf_day2"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 2 ET")
ax.plot(df["L3rnn_day2"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 2 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("Three days lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################################
## model 3
### lead time 3 day 3

####################################################################################################3
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L3rf_day3"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 3 ET")
ax.plot(df["L3rnn_day3"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 3 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("Three days lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()
##########################################################################################################################


######combine


plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
#ax.plot(x,y,'i', color="orange",markersize=2)
ax.plot(df["L3rf_day3"],'o',color="magenta",linewidth=0.3,linestyle="--",markersize=2.5,label="rf Day 3 ET")
ax.plot(df["L3rnn_day3"],'*',color="indigo",linewidth=0.3,linestyle="--",markersize=4,label="LSTM Day 3 ET")

#ax.plot(x2,y2,'.', color="green",markersize=1,label="Silt Loam")
#ax.plot(x3,y3,'v', color="blue",markersize=0.5,label="Loam")
#ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.legend(loc='lower center',prop={'size': 6})
ax.set_title("Three days lead time Input",size=8)
#ax.set_xlabel('RMSEZ (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('RMSE (mm)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
plt.xticks([0, 1, 2,3,4,5,6], ['Apr', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(8) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 7,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(10,40)
#ax.set_xlim(0,300)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()







###############################################################################
   
df=pd.read_csv('RMSE_forcast.csv')
df = df[(df['stat']=='ts')]
#plt.plot(df.rnnL2ET3*25.4,color='black')
#plt.plot(df.rmseforL1rfET3*25.4,color='orange')

#plt.plot(df.rmseforL1rfET1*25.4,color='r')
#plt.plot(df.rmseforL1rfET2*25.4,color='black')
#plt.plot(df.rmseforL1rfET3*25.4,color='orange')
df_mm=df.avg_rmseforL3rnnET*25.4

#df_mm=df.avg_rmseforL3rfET*25.4
buildingdf=pd.read_csv('RMSE_forcast.csv')
lat = buildingdf['lat'].values
long = buildingdf['long'].values
# read in data to use for plotted points
##buildingdf = pd.read_csv('RMSE_forcast.csv')  use this line if includinf training data too
buildingdf =df
lat = buildingdf['lat'].values
long = buildingdf['long'].values
#margin = 0 # buffer to add to the range
lat_min = min(lat) - 0.5
lat_max = max(lat) + 0.5
long_min = min(long) - 1
long_max = max(long) + 1
bbox = [39,45,-97,-82]


plt.rcParams['figure.figsize'] = (3.5,2.5)
#plt.rcParams['figure.figsize'] = (4.5,3.5)
fig, ax = plt.subplots()
#plt.rcParams['figure.figsize'] = (3.5,2.5)
plt.rcParams.update({'figure.autolayout': True})
#fig, ax = plt.subplots()

m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
            llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
m.drawcoastlines()
m.fillcontinents(color='white',lake_color='white')
m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0],linewidth=0.25,fontsize=7)
m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45,fontsize=7)
m.drawmapboundary(fill_color='blue')
source = ColumnDataSource(data = dict(lat = lat,lon = long))
state_lats = [states[code]["lats"] for code in states]
state_longs = [states[code]["lons"] for code in states]
m.drawcountries()
m.drawstates()
xpt,ypt = long,lat
#sc=m.scatter(xpt, ypt, c=df.avg_rmseforL3rnnET,marker='o',s=df.avg_rmseforL3rnnET*70,zorder=10,latlon=True)

sc=m.scatter(xpt, ypt,c=df_mm, cmap='viridis',vmin=10, vmax=30,marker='o',s=30,zorder=20,latlon=True)
#sc=m.scatter(xpt, ypt,c=df_mm, cmap='viridis',vmin=0, vmax=38,marker='o',s=df.avg_rmseforL3rnnET*70,zorder=10,latlon=True)
#ax.yaxis.label.set_size(5)
#sc.ax.tick_params(labelsize=10)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
#cax.ax.tick_params(labelsize=10)
plt.colorbar(sc,cax)
#plt.clim(0, 1.5)
#plt.colorbar(sc,cax)
#plt.tick_params(size=0)
#ax.set_title("3 Days ET Forecast Model: rf_forc,Lead Time:24,48 and 72 hours",size=6)
ax.set_title("3 Days ET Forecast Model: LSTM_forc,Lead Time:24,48 and 72 hours",size=5)

#plt.tick_params(axis = 'y', which = 'major', labelsize = 1,direction='in')
#plt.tick_params(axis = 'x', which = 'major', labelsize = 1,direction='in')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")

#####################################################################################################




import matplotlib as mpl
import matplotlib.cm as cm


import numpy as n
import matplotlib.pyplot as plt

# Random gaussian data.
plt.rcParams['figure.figsize'] = (1.5,1)
fig, ax = plt.subplots()
data = df.avg_rmseforL3rnnET*25.4

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('viridis')
#cm=cm(vmin=0, vmax=1.5)
# Plot histogram.
n, bins, patches = plt.hist(data, 4, normed=1)
#bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
#col = bin_centers - min(bin_centers)
col = data

#ax.set_title("Median 0.74",size=5)
#ax.set_title("Median RMSE 22.8 mm",size=5)  # rf
ax.set_title("Median 24.3",size=5)  # LSTM

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
ax=plt.gca()
ax.set_ylim(0,4)
ax.set_xlim(0,30)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')

fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

plt.show()

##########################################################################################################

########################################################################################################
### if error in basemap use this
####################################################################################################3




### for skill score
###https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/












