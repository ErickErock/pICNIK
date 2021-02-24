#!/usr/bin/python3.7
import numpy as np
import pandas as pd


class DataExtraction(object):

    def __init__(self):
        """
        Constructor. 
        Does not receive parameters 
        and ony establishes variables.
        """
        self.DFlis          = []
        self.Beta           = []
        self.BetaPC         = []
        self.files          = []
        self.dadt   	    = []
		self.T    	        = []
		self.t  	        = []
		self.Iso_convDF     = pd.DataFrame([],columns = []) 

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
    
		dadt        = self.dadt
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
		colnames = colnames[1:] + colnames[0]
		da_dt = da_dt[1:] + da_dt[0]
		T = T[1:] + T[0]
		t = t[1:] + t[0]
		Iso_convDF = Iso_convDF[colnames]  
		
		self.dadt   	  = dadt
		self.T      	  = T
		self.t      	  = t
	 	self.Iso_convDF   = Iso_convDF 	

