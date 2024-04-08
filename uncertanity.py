# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:30:58 2019

@author: Ammara
"""
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

import pandas as pd
import seaborn as sns


os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
#df=pd.read_csv('fore_test.csv')
df=pd.read_csv('fore_test_uncer.csv')




s1 = pd.Series(y1, index=x1).rename('s1')
s2 = pd.Series(y2, index=x2).rename('s2')

df = pd.concat([s1, s2], axis=1)

# Now let's unstack the dataframe so seaborn can recognize it
data = df.unstack().dropna().to_frame()
data.columns = ['values']



ax = sns.lineplot(x='level_1', y = 'values', hue='level_0',
              data=data.reset_index())

# Fill the missing points using interpolation
df_filled = df.copy().interpolate()

ma = df_filled.mean(axis=1).interpolate()

ax.plot(ma.index, ma, color='r', linestyle='--', label='mean')

mstd = ma.std()

ax.fill_between(ma.index, ma + mstd, ma - mstd,
                color='b', alpha=0.2)
plt.legend()














