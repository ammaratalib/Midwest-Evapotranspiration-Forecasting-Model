# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:51:48 2019

@author: Ammara
"""


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

### merge both together

   #    'avg_rmseforL1rfET',  'avg_rmseforL2rfET', 'avg_rmseforL3rfET',
   #    'avg_rmseforL1rnnET', 'avg_rmseforL2rnnET', , 'avg_rmseforL3rnnET'],
   #   dtype='object')
df=pd.read_csv('RMSE_forcast.csv')
buildingdf=pd.read_csv('RMSE_forcast.csv')
lat = buildingdf['lat'].values
long = buildingdf['long'].values
# read in data to use for plotted points
buildingdf = pd.read_csv('RMSE_forcast.csv')

lat = buildingdf['lat'].values
long = buildingdf['long'].values
#margin = 0 # buffer to add to the range
lat_min = min(lat) - 0.5
lat_max = max(lat) + 0.5
long_min = min(long) - 1
long_max = max(long) + 1
bbox = [39,45,-97,-82]



#plt.rcParams['figure.figsize'] = (3.5,2.5)
plt.rcParams['figure.figsize'] = (4.5,2.5)
plt.rcParams.update({'figure.autolayout': True})
fig, ax = plt.subplots()

m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
            llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
m.drawcoastlines()
m.fillcontinents(color='white',lake_color='white')
m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0],linewidth=0.25,fontsize=10)
m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45,fontsize=10)
m.drawmapboundary(fill_color='blue')
source = ColumnDataSource(data = dict(lat = lat,lon = long))
state_lats = [states[code]["lats"] for code in states]
state_longs = [states[code]["lons"] for code in states]
m.drawcountries()
m.drawstates()
xpt,ypt = long,lat
#sc=m.scatter(xpt, ypt, c=df.avg_rmseforL3rnnET,marker='o',s=df.avg_rmseforL3rnnET*70,zorder=10,latlon=True)
sc=m.scatter(xpt, ypt,c=df.avg_rmseforL3rnnET, cmap='viridis',vmin=0, vmax=1.5,marker='o',s=df.avg_rmseforL3rnnET*70,zorder=10,latlon=True)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc,cax)
#plt.clim(0, 1.5)
#plt.colorbar(sc,cax)
#plt.tick_params(size=0)
#ax.set_title("3 Days ET Forecast Model: rf_forc,Lead Time:24,48 and 72 hours",size=8)
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
data = df.avg_rmseforL3rnnET

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
#ax.set_title("Median 0.74",size=5)  # rf
ax.set_title("Median 0.92",size=5)  # LSTM

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c-0.3))
ax=plt.gca()
ax.set_ylim(0,4)
ax.set_xlim(0,1.5)
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

import skill_metrics as sm

plt.rcParams['figure.figsize'] = (8,10)
fig, ax = plt.subplots()
sm.taylor_diagram(0.22,0.52,0.32)




sm.target_diagram(0.22,0.52,0.32, ticks = np.arange(-50,60,10),alpha=1)
plt.show()

#https://makersportal.com/blog/2018/7/20/geographic-mapping-from-a-csv-file-using-python-and-basemap

# option 5, 7, 8,  2 for cone

sm.target_diagram()
sm.target_diagram(0.22,0.52,0.32, markerLabel = 'marker', ticks = np.arange(-50,60,10),alpha=5.0)



#https://github.com/PeterRochford/SkillMetrics/wiki/Target-Diagram-Options



sm.target_diagram(0.22,0.52,0.32, markerLabel = 'marker',option=circlelinespec
, ticks = np.arange(-50,60,10))



sm.taylor_diagram(sdev,crmsd,ccoef, styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'observation')


###########################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pickle
import skill_metrics as sm
from sys import version_info

def load_obj(name):
    # Load object from file in pickle format
    if version_info[0] == 2:
        suffix = 'pkl'
    else:
        suffix = 'pkl3'

    with open(name + '.' + suffix, 'rb') as f:
        return pickle.load(f) # Python2 succeeds

class Container(object): 
    
    def __init__(self, pred1, pred2, pred3, ref):
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        self.ref = ref
        
if __name__ == '__main__':

    # Close any previously open graphics windows
    # ToDo: fails to work within Eclipse
    plt.close('all')
        
    # Read data from pickle file
    data = load_obj('target_data')

    # Calculate statistics for target diagram
    target_stats1 = sm.target_statistics(data.pred1,data.ref,'data')
    target_stats2 = sm.target_statistics(data.pred2,data.ref,'data')
    target_stats3 = sm.target_statistics(data.pred3,data.ref,'data')
     
    # Store statistics in arrays, making the fourth element a repeat of
    # the first.
    bias = np.array([target_stats1['bias'], target_stats2['bias'], 
                     target_stats3['bias'], target_stats1['bias'],
                     0.991*target_stats3['bias']])
    crmsd = np.array([target_stats1['crmsd'], target_stats2['crmsd'], 
                      target_stats3['crmsd'], target_stats1['crmsd'],
                      target_stats3['crmsd']])
    rmsd = np.array([target_stats1['rmsd'], target_stats2['rmsd'], 
                     target_stats3['rmsd'], target_stats1['rmsd'],
                     target_stats3['rmsd']])

    # Specify labels for points in a list (M1 for model prediction # 1, 
    # etc.).
    label = ['M1', 'M2', 'M3', 'M4', 'M5']
    
    # Check for duplicate statistics
    duplicateStats = sm.check_duplicate_stats(bias,crmsd)
     
    # Report duplicate statistics, if any. 
    sm.report_duplicate_stats(duplicateStats)

    '''
    Produce the target diagram
    Label the points and change the axis options. Increase the upper limit
    for the axes, change color and line style of circles. Increase
    the line width of circles. Change color of labels and points. Add a
    legend.
    For an exhaustive list of options to customize your diagram, 
    please call the function at a Python command line:
    >> target_diagram
    '''
    
    #ToDo: fix placement of legend 
    sm.target_diagram(bias,crmsd,rmsd, markerLabel = label, \
                      markerLabelColor = 'b', \
                      markerColor = 'b', markerLegend = 'on', \
                      ticks = np.arange(-50,60,10), \
                      axismax = 50.0, \
                      circles = [20, 40, 50], \
                      circleLineSpec = 'b-.', circleLineWidth = 1.5,
                      markerSize = 10, alpha = 0.0)

    # Write plot to file
    plt.savefig('target7.png')

    # Show plot
    plt.show()


