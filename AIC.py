# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:53:05 2020

@author: Ammara
"""


from math import log
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
# calculate aic for regression



os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

def crop(df):
    if (df['crop'] =='corn'):
        return 'corn'
    elif (df['crop'] =='soy'):
        return 'soy'
    elif (df['crop'] =='potat1'):
        return 'potat'
    elif (df['crop'] =='potat2'):
        return 'potat'

def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic


def AIC_irri(g):            
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    pot_lomsarf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)


#-476.05 LSTM 16
#-1350.09 LSTM 11
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')

df['soil'] = df.apply(soil_df,axis=1)
df['crop'] = df.apply(crop,axis=1)

df_irr=df[(df['irr_nonirr']==1)] #irrigated
df_nonirr=df[(df['irr_nonirr']==0)] #irrigated
pot_lomsa=df[(df['soil']=='loamy sand') & (df['crop']=='potat')]
corn_siltL=df[(df['soil']=='silt loam') & (df['crop']=='corn')]
soy_siltL=df[(df['soil']=='silt loam') & (df['crop']=='soy')]

###  irrigated
def AIC_irri( g):
            
    df=pot_lomsa
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    pot_lomsarf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)

  
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    pot_lomsarf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    pot_lomsarf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    pot_lomsalst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    pot_lomsalst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    pot_lomsalst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    df=corn_siltL
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    corn_lmrf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    corn_lmrf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    corn_lmrf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    corn_lmlst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    corn_lmlst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    corn_lmlst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    df=soy_siltL
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    soy_siltLrf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    soy_siltLrf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    soy_siltLrf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    soy_siltLlst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    soy_siltLlst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    soy_siltLlst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)


 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return ([pot_lomsarf16,pot_lomsarf11,pot_lomsarf5,pot_lomsalst16,pot_lomsalst11, 
     pot_lomsalst5,corn_lmrf16,corn_lmrf11,corn_lmrf5,corn_lmlst16,corn_lmlst11,corn_lmlst5
     ,soy_siltLrf16,soy_siltLrf11,soy_siltLrf5,soy_siltLlst16,soy_siltLlst11,soy_siltLlst5])

AIC_ir=AIC_irri(df_irr)

###############################################################################################

###  
os.chdir(r"C:\ammara_MD\ML_ET\results_paper1")
df=pd.read_csv('pre_test.csv')
df['soil'] = df.apply(soil_df,axis=1)
df['crop'] = df.apply(crop,axis=1)
df=df[(df['irr_nonirr']==0)] 
corn_sanL=df[(df['soil']=='sandy loam') & (df['crop']=='corn')]
corn_lom=df[(df['soil']=='loam') & (df['crop']=='corn')]
soy_lom=df[(df['soil']=='loam') & (df['crop']=='soy')]
corn_siltL=df[(df['soil']=='silt loam') & (df['crop']=='corn')]
soy_siltL=df[(df['soil']=='silt loam') & (df['crop']=='soy')]

def AIC_nonirr( g):   
    df=corn_sanL
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    corn_sanrf16 = calculate_aic(len(df["test_obs16"]), mse, num_params) 
    
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    corn_sanrf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    corn_sanrf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    corn_sanlst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    corn_sanlst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    corn_sanlst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 

    df=corn_lom
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    corn_lmrf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    corn_lmrf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    corn_lmrf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    corn_lmlst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    corn_lmlst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    corn_lmlst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    df=soy_lom
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    soy_lmrf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    soy_lmrf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    soy_lmrf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    soy_lmlst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    soy_lmlst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    soy_lmlst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    df=corn_siltL
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    corn_siltLrf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    corn_siltLrf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    corn_siltLrf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    corn_siltLlst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    corn_siltLlst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    corn_siltLlst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    df=soy_siltL
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred16"])
    soy_siltLrf16 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred11"])
    soy_siltLrf11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rftest_pred5"])
    soy_siltLrf5 = calculate_aic(len(df["test_obs16"]), mse, num_params)
    
    num_params=16+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred16"])
    soy_siltLlst16 = calculate_aic(len(df["test_obs16"]), mse, num_params)
 
    num_params=11+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred11"])
    soy_siltLlst11 = calculate_aic(len(df["test_obs16"]), mse, num_params)

    num_params=5+2
    mse=mean_squared_error(df["test_obs16"], df["rnntest_pred5"])
    soy_siltLlst5 = calculate_aic(len(df["test_obs16"]), mse, num_params)

 #   return pd.Series( dict(  r2 = r2, rmse = rmse ) )
    return ([corn_sanrf16,corn_sanrf11,corn_sanrf5,corn_sanlst16,corn_sanlst11,corn_sanlst5,
             corn_lmrf16,corn_lmrf11,corn_lmrf5,corn_lmrf16,corn_lmlst11,corn_lmlst5,
             soy_lmrf16, soy_lmrf11 ,soy_lmrf5,soy_lmlst16,soy_lmlst11,soy_lmlst5,
corn_siltLrf16,corn_siltLrf11,corn_siltLrf5,corn_siltLlst16,corn_siltLlst11,corn_siltLlst5,
soy_siltLrf16,soy_siltLrf11,soy_siltLrf5,soy_siltLlst16, soy_siltLlst11,soy_siltLlst5])

AIC_nonirri=AIC_nonirr(df_nonirr)

dd = pd.DataFrame(np.array(AIC_nonirri).reshape(30,1), columns = list("s"))

dd["crop"]=["corn","corn","corn","corn","corn","corn","corn","corn","corn","corn","corn","corn",
  "soy","soy","soy","soy","soy","soy","corn","corn","corn","corn","corn","corn","soy","soy","soy","soy","soy","soy"]

dd["soil"]=["sandy Loam","sandy Loam","sandy Loam","sandy Loam","sandy Loam","sandy Loam","Loam","Loam","Loam","Loam","Loam","Loam"
  ,"Loam","Loam","Loam","Loam","Loam","Loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam"]

dd["model"]=["rf16","rf11","rf5","lstm16","lstm11","lstm5","rf16","rf11","rf5","lstm16","lstm11","lstm5"
,"rf16","rf11","rf5","lstm16","lstm11","lstm5","rf16","rf11","rf5","lstm16","lstm11","lstm5"
 , "rf16","rf11","rf5","lstm16","lstm11","lstm5"]

dd_non_irr=dd  #20 

############################################################################################
#irrigated
AIC_ir=AIC_irri(df_irr)
dd = pd.DataFrame(np.array(AIC_ir).reshape(18,1), columns = list("s")) 
dd["crop"]=["potat","potat","potat","potat","potat","potat","corn","corn","corn","corn","corn","corn","soy","soy","soy","soy","soy","soy"]
dd["model"]=["rf16","rf11","rf5","lstm16","lstm11","lstm5", "rf16","rf11","rf5","lstm16","lstm11","lstm5","rf16","rf11","rf5","lstm16","lstm11","lstm5"]  
dd["soil"]=["Loamy Sand","Loamy Sand","Loamy Sand","Loamy Sand","Loamy Sand","Loamy Sand","silt loam","silt loam","silt loam","silt loam", "silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam","silt loam"]
dd_irr=dd# 12

#############################################################################################
### overall AIC
df_s=pd.DataFrame(pd.concat((dd_non_irr['s'],dd_irr['s']),axis=0))
df_mod=pd.DataFrame(pd.concat((dd_non_irr['model'],dd_irr['model']),axis=0))
df=pd.DataFrame(pd.concat((df_s['s'],df_mod['model']),axis=1))
df=df.groupby("model").mean()
df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf5,rf 16,lstm 11,lstm 5,lstm 16

ff_overall=pd.DataFrame([0,0.67,0.153,0.22,1,0.80])
plt.rcParams['figure.figsize'] = (2.0, 1.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('AIC', color='black')
ax1.set_xlabel('ET prediction Models', color='black')
ff_overall=pd.DataFrame([0,0.67,0.153,0.22,1,0.80])
df=ff_overall
#plt.plot(ff_overall,linestyle="-",color='blue',markersize=2,marker='s')
plt.plot(ff_overall,color='blue',linestyle='--',linewidth=0.5,markersize=4,marker='s')

#ax1.legend()
#lgd=ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 4.5})
ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
ax1.set_ylim(0,1)
ax1.set_title("Growing Season (April-Oct)",fontsize=6)
ax1.set_xticklabels(('rf_11','rf_5','rf_16','LSTM_11','LSTM_5','LSTM_16'))
ax1.set_xticks(np.arange(0,6))
fig.autofmt_xdate() 

plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

###############################################################################################

### irrigated _non irrigated
dd_non_irr["irr_non"]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
dd_irr["irr_non"]=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

df_s=pd.DataFrame(pd.concat((dd_non_irr['s'],dd_irr['s']),axis=0))

df_mod=pd.DataFrame(pd.concat((dd_non_irr['model'],dd_irr['model']),axis=0))
df_irrno=pd.DataFrame(pd.concat((dd_non_irr['irr_non'],dd_irr['irr_non']),axis=0))

df=pd.DataFrame(pd.concat((df_s['s'],df_mod['model'],df_irrno["irr_non"]),axis=1))
df = df.groupby(['model','irr_non']).agg({'s':'mean'}).reset_index()
df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf 16,lstm 11,lstm 16

ff_irr=pd.DataFrame([0.16,0.86,0.44,0.32,0.88,1])
ff_nonirr=pd.DataFrame([0.0,0.44,0.03,0.18,0.85,0.51])


plt.rcParams['figure.figsize'] = (2.0, 1.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('AIC', color='black')
ax1.set_xlabel('ET prediction Models', color='black')
# rf11, rf 16,lstm 11,lstm 16
ff_overall=pd.DataFrame([0,0.047,0.27,1])
plt.plot(ff_irr,linestyle="-",color='black',label="Irrigated",markersize=2,marker='s')
plt.plot(ff_nonirr,linestyle='None',color='green',label="Non-Irrigated",markersize=2,marker='p')
ax1.legend()
lgd=ax1.legend(loc='bottom right',prop={'size': 4.5})
ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
ax1.set_ylim(0,1)
ax1.set_title("Growing Season (April-Oct)",fontsize=6)
ax1.set_xticklabels( ('rf_11','rf_5','rf_16','LSTM_11','LSTM_5','LSTM_16'))
ax1.set_xticks(np.arange(0,6))
fig.autofmt_xdate() 

plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

##############################################################################################

###############################################################################################


####################################################################################################

### irrigated _non irrigated
dd_non_irr["irr_non"]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
dd_irr["irr_non"]=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

df_s=pd.DataFrame(pd.concat((dd_non_irr['s'],dd_irr['s']),axis=0))

df_mod=pd.DataFrame(pd.concat((dd_non_irr['model'],dd_irr['model']),axis=0))
df_irrno=pd.DataFrame(pd.concat((dd_non_irr['irr_non'],dd_irr['irr_non']),axis=0))
df_soil=pd.DataFrame(pd.concat((dd_non_irr['soil'],dd_irr['soil']),axis=0))
df_crop=pd.DataFrame(pd.concat((dd_non_irr['crop'],dd_irr['crop']),axis=0))
df=pd.DataFrame(pd.concat((df_s['s'],df_mod['model'],df_irrno["irr_non"],df_soil["soil"],df_crop["crop"]),axis=1))
df = df.groupby(['model','crop','soil','irr_non']).agg({'s':'mean'}).reset_index()


corn_sanL=df[(df['soil']=='sandy Loam') & (df['crop']=='corn')]
corn_siltL=df[(df['soil']=='silt loam') & (df['crop']=='corn')]
corn_lom=df[(df['soil']=='Loam') & (df['crop']=='corn')]
pot_lomsa=df[(df['soil']=='Loamy Sand') & (df['crop']=='potat')]
soy_lom=df[(df['soil']=='Loam') & (df['crop']=='soy')]
soy_siltL=df[(df['soil']=='silt loam') & (df['crop']=='soy')]


ax.plot(a['avg_ET3']*25.4,a['ET3_obs']*25.4,'o', color="purple",markersize=2, fillstyle='none',label="Corn (Sandy Loam)")
ax.plot(c['avg_ET3']*25.4,c['ET3_obs']*25.4,'p', color="seagreen",markersize=2,fillstyle='none',label="Corn (Loam)")
ax.plot(d['avg_ET3']*25.4,d['ET3_obs']*25.4,'s', color="gold",markersize=2,fillstyle='none',label="Soybeans (Loam)")
ax.plot(e['avg_ET3']*25.4,e['ET3_obs']*25.4,'h', color="red",markersize=2,fillstyle='none',label="Corn (Silt Loam)")
ax.plot(f['avg_ET3']*25.4,f['ET3_obs']*25.4,'v', color="brown",markersize=2,fillstyle='none',label="Soybeans (Silt Loam)")
#ax.plot(b['avg_ET3']*25.4,b['ET3_obs']*25.4,'>', color="black",markersize=,fillstyle='none',label="Potatoes (Loamy Sand)")


plt.rcParams['figure.figsize'] = (2.0, 1.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel('AIC', color='black')
ax1.set_xlabel('ET prediction Models', color='black')
# rf11, rf 16,lstm 11,lstm 16

df=corn_sanL
df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf 16,lstm 11,lstm 16
df=pd.DataFrame([0,0.32,0.54,0.63,1,0.52]) # corn sand loam
#plt.plot(ff_irr,linestyle="-",color='purple',label="Corn (Sandy Loam)",markersize=2,marker='o')


df=corn_siltL
df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf 16,lstm 11,lstm 16
df=pd.DataFrame([0,0.79,0.48,0.1,0.89,1]) # corn silt loam
#plt.plot(df,linestyle="-",color='red',label="Corn (Silt Loam)",markersize=2,marker='h')

df=corn_lom
df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf 16,lstm 11,lstm 16
df=pd.DataFrame([0.27,0.39,0.0,0.46,1,0]) # corn silt loam
#plt.plot(df,linestyle="-",color='seagreen',label="Corn (Loam)",markersize=2,marker='p')


df=pot_lomsa
df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf 16,lstm 11,lstm 16
df=pd.DataFrame([0.0,0.78,0.32,1,0.18,0.4]) # corn silt loam
#plt.plot(df,linestyle="-",color='black',label="Potatoes (Loamy Sand)",markersize=2,marker='>')


df=soy_lom
df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf 16,lstm 11,lstm 16
#df=pd.DataFrame([0.31,0.39,0.0,0.46,1.0,0.47]) # corn silt loam

#plt.plot(df,linestyle="-",color='gold',label="Soybeans (Loam)",markersize=2,marker='s')

df=soy_siltL

df["normAIC"]=(df["s"]-min(df["s"]))/(max(df["s"])-min(df["s"]))
# rf11, rf 16,lstm 11,lstm 16
df=pd.DataFrame([0.0,0.77,0.18,0.11,0.85,1]) # corn silt loam

plt.plot(df,linestyle="-",color='brown',label="Soybeans (Silt Laom)",markersize=2,marker='v')

ax1.legend()
#lgd=ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 4.5})
lgd=ax1.legend(loc='upper center',prop={'size': 4.5})

ax1.yaxis.label.set_size(6)
ax1.xaxis.label.set_size(6) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 6)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 5)
ax1.set_ylim(0,1)
ax1.set_title("Growing Season (April-Oct)",fontsize=6)
ax1.set_xticklabels( ('rf_11','rf_5','rf_16','LSTM_11','LSTM_5','LSTM_16'))
ax1.set_xticks(np.arange(0,6))
fig.autofmt_xdate() 

plt.show()
fig.savefig('testfig1.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

####################################################################################################


























