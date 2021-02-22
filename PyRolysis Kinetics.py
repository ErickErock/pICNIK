#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.special as sp
from scipy.optimize import minimize_scalar

# In[2]:


#Lista: de los archivos en orden ascendente de HR (Heating Rate)

files2 = ['SAR 5 N2.csv','SAR 10  N2.csv','SAR 20 N2.csv','SAR 40  N2.csv','SAR 50  N2.csv'] 


# In[3]:


DFlis = []    #Lista: de DF de cada archivo en files
Beta = []     #Lista: de HR en el mismo orden en que están los archivos en files
BetaPearsCoeff= []    #Lista: de r^{2} para cada curva experimental T vs t


# In[4]:


#Función: Crea un pandas DataFrame (DF) de cada archivo en files. En cada DF crea las columnas 'Temperature [K]', 'alpha'

def data_extraction(filelist):
     for item in filelist:
        DF = pd.read_table(item, engine = 'python')
        DF['Temperature [K]'] = DF['Temperature (°C)'] + 273.15
        DF['alpha'] = (DF['Weight (mg)'][0]-DF['Weight (mg)'])/(DF['Weight (mg)'][0]-DF['Weight (mg)'][DF.shape[0]-1])
       
        y = DF['Temperature [K]']    
        x = DF['Time (min)']
        den = np.sum((x-(np.mean(x)))**2)
        num = np.sum((x-(np.mean(x)))*(y-(np.mean(y)))) 
        r = (num**2)/(np.sum((x-np.mean(x))**2)*(np.sum((y-np.mean(y))**2)))
        
        BetaPearsCoeff.append(r)
        Beta.append(num/den ) 
        DFlis.append(DF)


# In[5]:


data_extraction(files2)    #Ejecuta la función data_extraction sobre la lista files




# In[8]:


Iso_convDF = pd.DataFrame([],columns=[])  #DF: Nuevo DF con los datos de isoconversión
da_dt = []    #lista: de las derivadas de alpha respecto a t para cada HT
T = []    #Lista: de temperaturas que serán graficadas vs da_dt
t = []

#Función: Toma la DF con menos datos (que siempre corresponde al mayor HT, o sea, el último elemento de las DF's), filtra los valorees de 0.05 < alpha <0.95

def isoconversional(dflist):
    dflist[-1] = np.round(dflist[-1].loc[(dflist[-1]['alpha'] >= 0.05) & (dflist[-1]['alpha'] <= 0.95)], decimals = 7) 
    Iso_convDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(dflist[-1]['Temperature [K]'], decimals = 4)
    alps = dflist[-1]['alpha'].values
    da = []
    Ti = []
    ti = []
    for r in range(len(dflist[-1]['alpha'].values)-1):
        da.append((dflist[-1]['alpha'].values[r+1]-dflist[-1]['alpha'].values[r])/(dflist[-1]['Time (min)'].values[r+1]-dflist[-1]['Time (min)'].values[r]))
        Ti.append(dflist[-1]['Temperature [K]'].values[r])
        ti.append(dflist[-1]['Time (min)'].values[r])
    da_dt.append(da)
    T.append(Ti)
    t.append(ti)
    for i in range(0,len(dflist)-1): 
        dflist[i] = np.round(dflist[i].loc[(dflist[i]['alpha'] > 0.05) & (dflist[i]['alpha'] < 0.95)], decimals=7)
        inter_func = interp1d(dflist[i]['alpha'], dflist[i]['Temperature [K]'],kind='cubic', bounds_error=False, fill_value="extrapolate")
        Iso_convDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)
        db = []
        Tv=[]
        tv = []
        for r in range(len(dflist[i]['alpha'].values)-1):
            db.append((dflist[i]['alpha'].values[r+1]-dflist[i]['alpha'].values[r])/(dflist[i]['Time (min)'].values[r+1]-dflist[i]['Time (min)'].values[r]))
            Tv.append(dflist[i]['Temperature [K]'].values[r])
            tv.append(dflist[i]['Time (min)'].values[r])
        da_dt.append(db)
        T.append(Tv)
        t.append(tv)
    Iso_convDF.index = dflist[-1]['alpha'].values     

#%%
isoconversional(DFlis)
colnames = Iso_convDF.columns.tolist()
colnames = colnames[1:] + colnames[:1]
da_dt = da_dt[1:] + da_dt[:1]
T = T[1:] + T[:1]
t = t[1:] + t[:1]
Iso_convDF = Iso_convDF[colnames]
# In[9]:


AdvIsoDF = pd.DataFrame([],columns=[])

def adv_isoconversional(dflist):
    alps =  np.linspace(dflist[-1]['alpha'].values[0],dflist[-1]['alpha'].values[-1],Iso_convDF.shape[0]+1)
    for i in range(0,len(dflist)): 
        dflist[i] = np.round(dflist[i].loc[(dflist[i]['alpha'] > 0.05) & (dflist[i]['alpha'] < 0.95)], decimals=7)
        inter_func = interp1d(dflist[i]['alpha'], dflist[i]['Temperature [K]'],kind='cubic', bounds_error=False, fill_value="extrapolate")
        AdvIsoDF['HR '+str(np.round(Beta[i], decimals = 0)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)
    AdvIsoDF.index = alps     


# In[10]:

adv_isoconversional(DFlis)
# In[17]:


logB = np.log(Beta)
alpha = np.linspace(np.array(Iso_convDF.index)[0], np.array(Iso_convDF.index)[-1],18000)


E_FOW = []   

for i in range(0,Iso_convDF.shape[0]):  #Obtiene las energías de activación FOW
    y = (logB)    
    x = 1/(Iso_convDF.iloc[i].values)
    den = np.sum((x-(np.mean(x)))**2)
    num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
    E_a_i = -(8.314/1.052)*(num/den )*(1/1000)   
    E_FOW.append(E_a_i)

FOWinter = []
inter_func = interp1d(np.array(Iso_convDF.index), pd.Series(E_FOW),kind='cubic')
FOWinter.append(inter_func(alpha))

E_KAS = []
    
for i in range(0,Iso_convDF.shape[0]):      #Obtiene las energías de activación KAS
    y = (logB)- np.log((Iso_convDF.iloc[i].values)**1.92)    
    x = 1/(Iso_convDF.iloc[i].values)
    den = np.sum((x-(np.mean(x)))**2)
    num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
    E_a_i = -(8.314)*(num/den )*(1/1000)   
    E_KAS.append(E_a_i)

KASinter = []
inter_func = interp1d(np.array(Iso_convDF.index), pd.Series(E_KAS),kind='cubic') 
KASinter.append(inter_func(alpha))


E_vy = []

def omega(E, row,Tempdf = Iso_convDF):
    x = (1000*E)/(8.314*Tempdf.iloc[row])
    omega_i = []
    px = ((np.exp(-x))/x)*(((x**3)+(18*(x**2))+(88*x)+96)/((x**4)+(20*(x**3))+(120*(x**2))+(240*x)+120))
    p_B = px/Beta
    for j in range(len(Beta)):
        y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
        omega_i.append(y)
    O = np.array(np.sum((omega_i)))
    return O

for k in range(len(Iso_convDF.index)):
    E_vy.append(minimize_scalar(omega,args=(k),bounds=(-200,350),method = 'bounded').x)
 


def I(E,inf,up):
    a=E/8.314
    b=inf
    c=up
    
    return a*(sp.expi(-a/c)-sp.expi(-a/b)) + c*np.exp(-a/c) - b*np.exp(-a/b)


def adv_omega(E,rowi,Tempdf = AdvIsoDF):

    E = E*1000
    j = rowi
    omega_i = []
    I_x = []
    for i in range(len(AdvIsoDF.columns)):
        I_x.append(I(E,AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j]],AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j+1]]))  
    I_B = np.array(I_x)/Beta
    for k in range(len(Beta)):
        y = I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k]))
        omega_i.append(y)
    O = np.array(np.sum((omega_i)))
    return O

    
E_Vyz = []

for k in range(len(AdvIsoDF.index)-1):
    E_Vyz.append(minimize_scalar(adv_omega,bounds=(-300,500),args=(k), method = 'bounded').x)

interfunc = interp1d(AdvIsoDF.index.values[:-1], E_Vyz,kind='cubic', bounds_error=False, fill_value="extrapolate")
E_Vyz = interfunc(Iso_convDF.index)

#%%
def saving_results(dflist):
    DFreslis = []
    for k in range(len(Beta)):
        DF = pd.DataFrame([], columns=['Time [min]','Weight [mg]','Temperature [K]','alpha','da_dt'])
        DF['Time [min]'] = dflist[k]['Time (min)'].values
        DF['Weight [mg]'] = dflist[k]['Weight (mg)'].values
        DF['Temperature [K]'] = dflist[k]['Temperature [K]'].values
        DF['alpha'] = dflist[k]['alpha'].values
        DF['da_dt'][0] = np.nan
        DF['da_dt'][1:]=da_dt[k]
        
        DFreslis.append(DF)
        
    alps = Iso_convDF.index.values

    DF_nrgy = pd.DataFrame([], columns = ['FOW','KAS','Vyazovkin','Adv. Vyazovkin'])
    DF_nrgy['FOW']=E_FOW
    DF_nrgy['KAS'] = E_KAS
    DF_nrgy['Vyazovkin'] = E_vy
    DF_nrgy['Adv. Vyazovkin'] = E_Vyz
    DF_nrgy.index = alps     

    with pd.ExcelWriter('Activation_energies_results.xlsx') as writer:
        for i in range(len(DFreslis)):
            DFreslis[i].to_excel(writer, sheet_name='B ='+ str(np.round(Beta[i],decimals=1)) + 'K_min')
        DF_nrgy.to_excel(writer, sheet_name='Act. Energies')    
#%%
            
saving_results(DFlis)

#%%
import matplotlib.pyplot as plt

alps =  Iso_convDF.index.values

fig, ax = plt.subplots()  
ax.plot(alps, np.array(E_KAS),label='KAS',color='b') 
ax.plot(alps, np.array(E_FOW),label='FOW',color='m') 
ax.plot(alps, np.array(E_vy),label='Vyazovkin',color='g')
ax.plot(alps,E_Vyz,color='r',label='Adv. Vyazovkin')   
ax.set_xlabel(r"$\alpha$")  
ax.set_ylabel(r'$E_\alpha$')
ax.set_ylim(-350,500) 
ax.legend()
plt.xlim(0,1)


fig, ax = plt.subplots()    
for i in range(5):
    ax.plot(T[i], da_dt[i], label=str(np.round(Beta[i],decimals=0))+' K/min')
ax.set_xlabel('Temperature [K]')  
ax.set_ylabel(r"$\frac{d\alpha}{dt}$")  
    #ax.set_title("alpha prime vs Temperature [K]") 
ax.legend()
    #plt.xlim(0,75)

fig, ax = plt.subplots()  
for i in range(len(Beta)):
    ax.plot(DFlis[i]['Temperature [K]'], DFlis[i]['alpha'], label=str(np.round(Beta[i],decimals=0))+' K/min''')     
ax.set_xlabel('Temperature [K]')  
ax.set_ylabel(r'$\alpha$')  
#ax.set_title("Sargazo's Thermogravimetry") 
ax.legend()
