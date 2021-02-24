#!/usr/bin/python3.7
import numpy as np
import pandas as pd
from   scipy.interpolate import interp1d
from   scipy.optimize    import minimize_scalar
import scipy.special     as     sp

class DataExtraction(object):

    def __init__(self):
        """
        Constructor. 
        Does not receive parameters 
        and ony establishes variables.
        """
        self.DFlis      = [] 
        self.Beta       = []
        self.BetaPC     = []
        self.files      = []
        self.da_dt      = []
        self.T          = []
        self.t          = []
        self.Iso_convDF = pd.DataFrame([],columns = []) 
        self.AdvIsoDF   = pd.DataFrame([],columns=[])

    def set_datos(self, lista_archivos):
        """
        Method to establish the file list
        for the extrator
        """
        self.files = lista_archivos

    def data_extraction(self):
        """
        Method to extract the data contained in the files
        into a list of DataFrames. Adds a column
        corresponding to the Temperature in Kelvin and
        computes The heating rate (Beta) with 
        its Pearson Coefficient.
        """
        BetaPearsCoeff = self.BetaPC
        DFlis          = self.DFlis
        Beta           = self.Beta
        filelist       = self.files
        print("Archivos a ocupar: \n{} ".format(filelist))

        for item in filelist:

            DF = pd.read_table(item, engine = 'python', encoding="latin1")
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
        self.BetaPC = BetaPearsCoeff
        self.DFlis  = DFlis
        self.Beta   = Beta

    def isoconversional(self):
        """
        Method that builds a DataFrame based 
        on the isoconversional principle
        """

        da_dt       = self.da_dt
        T           = self.T
        t           = self.t
        dflist      = self.DFlis
        Iso_convDF  = self.Iso_convDF   
       
        dflist[-1] = np.round(dflist[-1].loc[(dflist[-1]['alpha'] >= 0.05) & (dflist[-1]['alpha'] <= 0.95)], decimals = 7) 
        Iso_convDF['HR '+str(np.round(self.Beta[-1], decimals = 1)) + ' K/min'] = np.round(dflist[-1]['Temperature [K]'], decimals = 4)
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
            Iso_convDF['HR '+str(np.round(self.Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)
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

        colnames = Iso_convDF.columns.tolist()
        colnames = colnames[1:] + colnames[:1]
        da_dt = da_dt[1:] + da_dt[:1]
        T = T[1:] + T[:1]
        t = t[1:] + t[:1]
        Iso_convDF = Iso_convDF[colnames]  
        
        self.da_dt   	  = da_dt
        self.T      	  = T
        self.t      	  = t
        self.Iso_convDF   = Iso_convDF 	

    def get_df_isoconv(self):
        """
        getter dataframe isoconversional 
        """
        return self.Iso_convDF

    def adv_isoconversional(self):
        """
        Method advanced isoconversional 
        """
        dflist     = self.DFlis
        AdvIsoDF   = self.AdvIsoDF
        Iso_convDF = self.Iso_convDF
        Beta       = self.Beta

        
        alps =  np.linspace(dflist[-1]['alpha'].values[0],dflist[-1]['alpha'].values[-1],Iso_convDF.shape[0]+1)
        for i in range(0,len(dflist)):
            dflist[i] = np.round(dflist[i].loc[(dflist[i]['alpha'] > 0.05) & (dflist[i]['alpha'] < 0.95)], decimals=7)
            inter_func = interp1d(dflist[i]['alpha'], 
                                  dflist[i]['Temperature [K]'],
                                  kind='cubic', bounds_error=False, 
                                  fill_value="extrapolate")
            AdvIsoDF['HR '+str(np.round(Beta[i], decimals = 0)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)
        AdvIsoDF.index = alps

        self.AdvIsoDF = AdvIsoDF
        self.DFlis    = dflist
        
    def get_adviso(self):
        """
        Getter for dataframe AdvIso
        """
        return self.AdvIsoDF

    def get_beta(self):
        """
        Getter for Beta
        """
        return self.Beta

    def get_dadt(self):
        """
        Getter for da_dt
        """
        return self.da_dt

    def get_t(self):
        """
        Getter for t
        """
        return self.t



class ActivationEnergy(object):
    """
    """

    def __init__(self, iso_df, Beta, adv_df=None):
        """
        """
        self.E_FOW = []
        self.E_KAS = []
        self.E_vy  = []
        self.E_vyz = []
        self.Iso_convDF = iso_df
        self.Beta  = Beta
        self.logB = np.log(Beta)
        self.alpha = np.linspace(np.array(iso_df.index)[0], 
                                 np.array(iso_df.index)[-1],18000)
        if(adv_df is not None):
            self.Adv_IsoDF = adv_df
        """ 
        Universal gas constant
        0.0083144626 kJ/(mol*K)
        """
        self.R =0.0083144626

    def FOW(self):
        """
        FOW method

        """
        logB       = self.logB
        E_FOW      = self.E_FOW
        Iso_convDF = self.Iso_convDF
        for i in range(0,Iso_convDF.shape[0]):  #Obtiene las energías de activación FOW
            y = (logB)
            x = 1/(Iso_convDF.iloc[i].values)
            den = np.sum((x-(np.mean(x)))**2)
            num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
            E_a_i = -(self.R/1.052)*(num/den)
            E_FOW.append(E_a_i)
        self.E_FOW = E_FOW

    def get_E_FOW(self):
        """
        Getter for E_FOW
        """
        return self.E_FOW

    def KAS(self):
        """
        """
        logB       = self.logB
        Iso_convDF = self.Iso_convDF
        E_KAS      = self.E_KAS
        for i in range(0,Iso_convDF.shape[0]):      #Obtiene las energías de activación KAS
            y = (logB)- np.log((Iso_convDF.iloc[i].values)**1.92)
            x = 1/(Iso_convDF.iloc[i].values)
            den = np.sum((x-(np.mean(x)))**2)
            num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
            E_a_i = -(self.R)*(num/den )
            E_KAS.append(E_a_i)
        self.E_KAS = E_KAS

    def get_E_KAS(self):
        """
        Getter for E_KAS
        """
        return self.E_KAS


    def vy(self):
        """
        """
        pass

    def vyz(self):
        """
        """
        pass
