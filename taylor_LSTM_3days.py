# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:15:10 2019

@author: Ammara
"""

def load_obj(name):
    # Load object from file in pickle format
    if version_info[0] == 2:
        suffix = 'pkl'
    else:
        suffix = 'pkl3'

    with open(name + '.' + suffix, 'rb') as f:
        return pickle.load(f) # Python2 succeeds
    
class Container(object): 
    def __init__(self, target_stats1, target_stats2, taylor_stats1, taylor_stats2):
        self.target_stats1 = target_stats1
        self.target_stats2 = target_stats2
        self.target_stats3 = target_stats3
    #    self.target_stats4 = target_stats4
     #   self.target_stats5 = target_stats5
      #  self.target_stats6 = target_stats6
    
if __name__ == '__main__':

    # Set the figure properties (optional)
    plt.rcParams["figure.figsize"] = [8.0, 6]
    plt.rcParams['lines.linewidth'] = 1 # line width for plots
    plt.rcParams.update({'font.size': 12}) # font size of axes text
    
    # Close any previously open graphics windows
    # ToDo: fails to work within Eclipse
    plt.close('all')
    
    # Read Taylor statistics for ERA Interim (stats1) and TRMM (stats2) 
    # data with respect to APHRODITE observations for each of years 2001 to 
    # 2014 from pickle file
#    stats = load_obj('Mekong_Basin_data') # observations

    # Specify labels for points in a dictionary because only desire labels
    # for each data set.
    taylor_stats1 = sm.taylor_statistics(df_MI_T3.rnnL3ET1*25.4,df_MI_T3.ET1_obs*25.4,'df_MI_T3')
    taylor_stats2 = sm.taylor_statistics(df_MI_T3.rnnL3ET2*25.4,df_MI_T3.ET2_obs*25.4,'df_MI_T3')
    taylor_stats3 = sm.taylor_statistics(df_MI_T3.rnnL3ET3*25.4,df_MI_T3.ET3_obs*25.4,'df_MI_T3')


    taylor_stats4 = sm.taylor_statistics(df_NB_Ne2.rnnL3ET1*25.4,df_NB_Ne2.ET1_obs*25.4,'df_NB_Ne2')
    taylor_stats5 = sm.taylor_statistics(df_NB_Ne2.rnnL3ET2*25.4,df_NB_Ne2.ET2_obs*25.4,'df_NB_Ne2')
    taylor_stats6 = sm.taylor_statistics(df_NB_Ne2.rnnL3ET3*25.4,df_NB_Ne2.ET3_obs*25.4,'df_NB_Ne2')

    taylor_stats7 = sm.taylor_statistics(df_MN_Ro3.rnnL3ET1*25.4,df_MN_Ro3.ET1_obs*25.4,'df_MN_Ro3')
    taylor_stats8 = sm.taylor_statistics(df_MN_Ro3.rnnL3ET2*25.4,df_MN_Ro3.ET2_obs*25.4,'df_MN_Ro3')
    taylor_stats9 = sm.taylor_statistics(df_MN_Ro3.rnnL3ET3*25.4,df_MN_Ro3.ET3_obs*25.4,'df_MN_Ro3')

    taylor_stats10 = sm.taylor_statistics(df_IA_Br1.rnnL3ET1*25.4,df_IA_Br1.ET1_obs*25.4,'df_IA_Br1')
    taylor_stats11 = sm.taylor_statistics(df_IA_Br1.rnnL3ET2*25.4,df_IA_Br1.ET2_obs*25.4,'df_IA_Br1')
    taylor_stats12 = sm.taylor_statistics(df_IA_Br1.rnnL3ET3*25.4,df_IA_Br1.ET3_obs*25.4,'df_IA_Br1')

    taylor_stats13 = sm.taylor_statistics(df_WI_CS.rnnL3ET1*25.4,df_WI_CS.ET1_obs*25.4,'df_WI_CS')
    taylor_stats14 = sm.taylor_statistics(df_WI_CS.rnnL3ET2*25.4,df_WI_CS.ET2_obs*25.4,'df_WI_CS')
    taylor_stats15 = sm.taylor_statistics(df_WI_CS.rnnL3ET3*25.4,df_WI_CS.ET3_obs*25.4,'df_WI_CS')

    taylor_stats16 = sm.taylor_statistics(df_MN_Ro1.rnnL3ET1*25.4,df_MN_Ro1.ET1_obs*25.4,'df_MN_Ro1')
    taylor_stats17 = sm.taylor_statistics(df_MN_Ro1.rnnL3ET2*25.4,df_MN_Ro1.ET2_obs*25.4,'df_MN_Ro1')
    taylor_stats18 = sm.taylor_statistics(df_MN_Ro1.rnnL3ET3*25.4,df_MN_Ro1.ET3_obs*25.4,'df_MN_Ro1')

    #label = {'MI_ET Day1': 'r', 'MI_ET Day2': 'b','MI_ET Day3': 'g'}
    #label = {'ET Day1': 'black', 'ET Day2': 'red','ET Day3': 'purple',}
    #label= ['14197', '14442', '14713', '14484', '14841', '15240', '15320', 
    label= ['14197', '14442', '14713'] 
    

    label={ i : label[i] for i in range(0, len(label) )} 

#        '15516', '15571', '15790', '15792', '15825', '15844', '16058', '16059', 
 #            '16060', '16066', '16091'] 
 
    
    '''
    Produce the Taylor diagram for the first dataset
    '''
    '''
    Overlay the second dataset
    '''
    sdev1 = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd1 = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef1 = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1]])


    sdev2 = np.array([taylor_stats2['sdev'][0], taylor_stats2['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd2 = np.array([taylor_stats2['crmsd'][0], taylor_stats2['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
    ccoef2 = np.array([taylor_stats2['ccoef'][0], taylor_stats2['ccoef'][1]])
     

    sdev3 = np.array([taylor_stats3['sdev'][0], taylor_stats3['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd3 = np.array([taylor_stats3['crmsd'][0], taylor_stats3['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef3 = np.array([taylor_stats3['ccoef'][0], taylor_stats3['ccoef'][1]])


    sdev4 = np.array([taylor_stats4['sdev'][0], taylor_stats4['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd4 = np.array([taylor_stats4['crmsd'][0], taylor_stats4['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef4 = np.array([taylor_stats4['ccoef'][0], taylor_stats4['ccoef'][1]])

    sdev5 = np.array([taylor_stats5['sdev'][0], taylor_stats5['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd5 = np.array([taylor_stats5['crmsd'][0], taylor_stats5['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef5 = np.array([taylor_stats5['ccoef'][0], taylor_stats5['ccoef'][1]])

    sdev6 = np.array([taylor_stats6['sdev'][0], taylor_stats6['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd6 = np.array([taylor_stats6['crmsd'][0], taylor_stats6['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef6 = np.array([taylor_stats6['ccoef'][0], taylor_stats6['ccoef'][1]])


    sdev7 = np.array([taylor_stats7['sdev'][0], taylor_stats7['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd7 = np.array([taylor_stats7['crmsd'][0], taylor_stats7['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef7 = np.array([taylor_stats7['ccoef'][0], taylor_stats7['ccoef'][1]])

    sdev8 = np.array([taylor_stats8['sdev'][0], taylor_stats8['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd8 = np.array([taylor_stats8['crmsd'][0], taylor_stats8['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef8 = np.array([taylor_stats8['ccoef'][0], taylor_stats8['ccoef'][1]])


    sdev9 = np.array([taylor_stats9['sdev'][0], taylor_stats9['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd9 = np.array([taylor_stats9['crmsd'][0], taylor_stats9['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef9 = np.array([taylor_stats9['ccoef'][0], taylor_stats9['ccoef'][1]])


    sdev10 = np.array([taylor_stats10['sdev'][0], taylor_stats10['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd10 = np.array([taylor_stats10['crmsd'][0], taylor_stats10['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef10 = np.array([taylor_stats10['ccoef'][0], taylor_stats10['ccoef'][1]])


    sdev11 = np.array([taylor_stats11['sdev'][0], taylor_stats11['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd11 = np.array([taylor_stats11['crmsd'][0], taylor_stats11['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef11 = np.array([taylor_stats11['ccoef'][0], taylor_stats11['ccoef'][1]])


    sdev12 = np.array([taylor_stats12['sdev'][0], taylor_stats12['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd12 = np.array([taylor_stats12['crmsd'][0], taylor_stats12['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef12 = np.array([taylor_stats12['ccoef'][0], taylor_stats12['ccoef'][1]])


    sdev13 = np.array([taylor_stats13['sdev'][0], taylor_stats13['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd13 = np.array([taylor_stats13['crmsd'][0], taylor_stats13['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef13 = np.array([taylor_stats13['ccoef'][0], taylor_stats13['ccoef'][1]])


    sdev14 = np.array([taylor_stats14['sdev'][0], taylor_stats14['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd14 = np.array([taylor_stats14['crmsd'][0], taylor_stats14['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef14 = np.array([taylor_stats14['ccoef'][0], taylor_stats14['ccoef'][1]])


    sdev15 = np.array([taylor_stats15['sdev'][0], taylor_stats15['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd15 = np.array([taylor_stats15['crmsd'][0], taylor_stats15['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef15 = np.array([taylor_stats15['ccoef'][0], taylor_stats15['ccoef'][1]])


    sdev16 = np.array([taylor_stats16['sdev'][0], taylor_stats16['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd16 = np.array([taylor_stats16['crmsd'][0], taylor_stats16['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef16 = np.array([taylor_stats16['ccoef'][0], taylor_stats16['ccoef'][1]])

    sdev17 = np.array([taylor_stats17['sdev'][0], taylor_stats17['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd17 = np.array([taylor_stats17['crmsd'][0], taylor_stats17['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef17 = np.array([taylor_stats17['ccoef'][0], taylor_stats17['ccoef'][1]])

    sdev18 = np.array([taylor_stats18['sdev'][0], taylor_stats18['sdev'][1]])
                     #,taylor_stats4['sdev'][1],taylor_stats5['sdev'][1],taylor_stats6['sdev'][1]])
        
    crmsd18 = np.array([taylor_stats18['crmsd'][0], taylor_stats18['crmsd'][1]])
                      #,taylor_stats4['crmsd'][1], taylor_stats5['crmsd'][1],taylor_stats6['crmsd'][1]])
       
    ccoef18 = np.array([taylor_stats18['ccoef'][0], taylor_stats18['ccoef'][1]])


    sm.taylor_diagram(sdev1,crmsd1,ccoef1, markerobs = '*',markerColor = 'black'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6))    

    sm.taylor_diagram(sdev2,crmsd2,ccoef1, markerobs = '*',markerColor = 'orange'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev3,crmsd3,ccoef3, markerobs = '*',markerColor = 'red'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev4,crmsd4,ccoef4, markerobs = '*',markerColor = 'black'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev5,crmsd5,ccoef5, markerobs = '*',markerColor = 'orange'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev6,crmsd6,ccoef6, markerobs = '*',markerColor = 'red'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev7,crmsd7,ccoef7, markerobs = '*',markerColor = 'black'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev8,crmsd8,ccoef8, markerobs = '*',markerColor = 'orange'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev9,crmsd9,ccoef9, markerobs = '*',markerColor = 'red'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev10,crmsd10,ccoef10, markerobs = '*',markerColor = 'black'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev11,crmsd11,ccoef11, markerobs = '*',markerColor = 'orange'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev12,crmsd12,ccoef12, markerobs = '*',markerColor = 'red'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev13,crmsd13,ccoef13, markerobs = '*',markerColor = 'black'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev14,crmsd14,ccoef14, markerobs = '*',markerColor = 'orange'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev15,crmsd15,ccoef15, markerobs = '*',markerColor = 'red'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev16,crmsd16,ccoef16, markerobs = '*',markerColor = 'black'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev17,crmsd17,ccoef17, markerobs = '*',markerColor = 'orange'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

    sm.taylor_diagram(sdev18,crmsd18,ccoef18, markerobs = '*',markerColor = 'red'
                       ,
                      markerSize = 20,
                      tickRMSangle = 110, showlabelsRMS = 'on',
                      titleRMS = 'off',tickSTD = np.arange(0,70,10),axismax=70.0,
                      colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
                      colCOR = 'k', styleCOR = '--', widthCOR = 1.0,tickRMS = np.arange(0,60,6),overlay='on')    

        # Write plot to file
    plt.xlabel('RMSD',color='g')  # we already handled the x-label with ax1
    #plt.title("rf 3 Days ET Forecast",size=20)

    plt.savefig('taylor12.png',dpi=150,facecolor='w', bbox_inches='tight')

    # Show plot
    plt.show()



# color by state
# change ET 1 2 or 3 by marker
    
    
    
    
    sm.target_diagram(bias2,crmsd2,rmsd2
                      , markerColor = 'r',markerLabel=label2,markerlegend='on'
                      ,ticks = np.arange(-60,70,10), axismax = 50, \
                      circleLineSpec = '-.b', circleLineWidth = 1.5,markerSize = 10,alpha=0,overlay='on')
