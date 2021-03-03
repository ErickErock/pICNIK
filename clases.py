#!/usr/bin/python3.7
import numpy as np
import pandas as pd
from   scipy.interpolate import interp1d
from   scipy.optimize    import minimize_scalar
import scipy.special     as     sp

class DataExtraction(object):
    """
    class that manipulates raw data
    to create lists and Data Frames 
    that will be used to compute the 
    Activation Energy
    """

    def __init__(self):
        """
        Constructor. 
        Does not receive parameters
        and only establishes variables.
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
        for the extrator. The list must be 
		in ascendent heating rate order.
        """
        self.files = lista_archivos

    def data_extraction(self):
        """
        Method to extract the data contained in the files
        into a list of DataFrames. Adds two columns: one
        corresponding to the Temperature in Kelvin and
        other corresponding to the conversion ('alpha').
        Also computes The heating rate ('Beta') with 
        its Pearson Coefficient.
        """
        BetaPearsCoeff = self.BetaPC
        DFlis          = self.DFlis
        Beta           = self.Beta
        filelist       = self.files
        print("Archivos a ocupar: \n{} ".format(filelist))

        for item in filelist:

            #ToDo: probar con utf8
            DF = pd.read_csv(item,  encoding="latin1", sep='\t')
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
        on the isoconversional principle by
		building a function interpolated from
		the data frame with least data points,
		wich corresponds to te fila with least 
		data points.
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
        Getter dataframe isoconversional. 
        """
        return self.Iso_convDF

    def adv_isoconversional(self):
        """
        Method that builds a DataFrame
		based on the advanced Vyazovkin 
		method. The index correspond to 
		an equidistant set of data from 
		the first calculated value in 
		the first element of DFlis.  
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

    def get_betaPC(self):
        """
        Getter for the Beta
        Pearson Coefficients
        """
        return self.BetaPC

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
    
    def get_T(self):
        """
        Getter for T
        """
        return self.T

    def get_valores(self):
        """
        Global getter for: Iso_convDF,
        AdvIsoDF, da_dt, t and Beta; 
        in that order.   
        """
        return [self.get_df_isoconv(), 
                self.get_adviso(),
                self.get_dadt(),
                self.get_T(),
                self.get_t(),
                self.get_beta()]

    def save_df(self, E_FOW, E_KAS, E_vy, E_Vyz, dialect="xlsx" ):
        """
        Method to save dataframe with
        values calculated by ActivationEnergy
        """
        Iso_convDF = self.Iso_convDF
        dflist     = self.DFlis
        Beta       = self.Beta
        da_dt      = self.da_dt
        T          = self.T
        t          = self.t

        DFreslis = []
        for k in range(len(Beta)):
            DF = pd.DataFrame([], columns=['time [min]',
                                           'Temperature [K]',
                                           'da_dt'])
            DF['time [min]'] = t[k]
            DF['Temperature [K]'] = T[k]
            DF['da_dt']=da_dt[k]

            DFreslis.append(DF)

        alps = Iso_convDF.index.values

        DF_nrgy = pd.DataFrame([], columns = ['alpha','FOW','KAS','Vyazovkin','Adv. Vyazovkin'])
        DF_nrgy['alpha']  = alps
        DF_nrgy['FOW']=E_FOW
        DF_nrgy['KAS'] = E_KAS
        DF_nrgy['Vyazovkin'] = E_vy
        DF_nrgy['Adv. Vyazovkin'] = E_Vyz

        if(dialect=='xlsx'):
            nombre = 'Activation_energies_results.xlsx'
            with pd.ExcelWriter(nombre) as writer:
                for i in range(len(DFreslis)):
                    DFreslis[i].to_excel(writer, 
                                         sheet_name='B ='+ str(np.round(Beta[i],decimals=1)) + 'K_min',
                                         index=False)
                DF_nrgy.to_excel(writer, sheet_name='Act. Energies',index=False)    

            print("Workseet {} ".format(nombre))
        elif(dialect=='csv'):
            print("Saving csvs\n")
            for i in range(len(Beta)):
                nombre = 'HR={0:0.3}_K_per_min.csv'.format(self.Beta[i])    
                df = pd.DataFrame({'t':t[i], 
                                   'T':T[i], 
                                   'da_dt':da_dt[i]})
                print("Saving {}".format(nombre))
                df.to_csv(nombre, sep=',',index=False)
            print("Saving activation energies")
            DF_nrgy.to_csv('Activation_energies_results.csv', 
                           encoding='utf8', 
                           sep=',',
                           index=False)


        else:
            print("Dialect not recognized")


class ActivationEnergy(object):
    """
	Class that uses the lists and
	Data Frames generated with 
	Dataextraction to compute 
	Activation energy values based on 
	four methods: FOW, KAS, Vyazovkin
	and Advanced Vyazovkin. 
    """

    def __init__(self, iso_df, Beta, adv_df=None):
        """
		Constructor. Receives the Isoconversional
		Data Frame as first parameter, the list 'Beta'
		as second parameter, and the Adv_Isoconversional
		Data Frame as an optional third parameter.
        """
        self.E_FOW = []
        self.E_KAS = []
        self.E_vy  = []
        self.E_Vyz = []
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
        Method to compute the Activation 
		Energy based on the Flynn-Osawa-Wall 
		(FOW) treatment.
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
        self.E_FOW = np.array(E_FOW)

    def get_E_FOW(self):
        """
        Getter for E_FOW
        """
        return self.E_FOW

    def KAS(self):
        """
        Method to compute the Activation 
		Energy based on the 
		Kissinger-Akahira-Sunose (KAS) treatment.
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
        self.E_KAS = np.array(E_KAS)

    def get_E_KAS(self):
        """
        Getter for E_KAS
        """
        return self.E_KAS

    def omega(self, E, row):
        """
        Function to minimize according
        to the Vyazovkin treatment.
        """
        Tempdf     = self.Iso_convDF
        Beta       = self.Beta

        x = E/(self.R*Tempdf.iloc[row])
        omega_i = []
        px = ((np.exp(-x))/x)*(((x**3)+(18*(x**2))+(88*x)+96)/((x**4)+(20*(x**3))+(120*(x**2))+(240*x)+120))
        p_B = px/Beta
        for j in range(len(Beta)):
            y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
            omega_i.append(y)
        O = np.array(np.sum((omega_i)))
        #ToDo: generar arreglo que contenga todos los valores O
        return O

    def set_bounds(self, bounds):
        """
        Setter for bounds variable
        """
        self.bounds = bounds

    def visualize_omega(self,row,N=1000):
        """
        Method to visualize omega function.
        Bounds requiered from function vy o 
        by bounds setter
        """
        bounds = self.bounds
        A = np.linspace(bounds[0], bounds[1], N)
        O = np.array([float(self.omega(A[i],row)) for i in range(len(A))])
        return A, O


    def vy(self, bounds):
        """
        Method to compute the Activation 
		Energy based on the Vyazovkin treatment.
        """
        E_vy       = self.E_vy
        Tempdf     = self.Iso_convDF
        Beta       = self.Beta 
        for k in range(len(Tempdf.index)):
            E_vy.append(minimize_scalar(self.omega, args=(k),bounds=bounds, method = 'bounded').x)

        self.E_vy = np.array(E_vy)

    def get_Evy(self):
        """
        Getter for E_Vy
        """
        return self.E_vy

    def vyz(self,bounds):
        """
        Method to compute the Activation 
		Energy based on the Advanced
		Vyazovkin treatment.
        """

        AdvIsoDF = self.Adv_IsoDF 
        Beta     = self.Beta

        def I(E,inf,up):
            a=E/(self.R)
            b=inf
            c=up

            return a*(sp.expi(-a/c)-sp.expi(-a/b)) + c*np.exp(-a/c) - b*np.exp(-a/b)
        
        
        def adv_omega(E,rowi,Tempdf = AdvIsoDF):
            """
            Function to minimize according
			to the advanced Vyazovkin treatment
            """
            j = rowi
            omega_i = []
            I_x = []
            for i in range(len(AdvIsoDF.columns)):
                I_x.append(I(E,
                             AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j]],
                             AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j+1]]))
            I_B = np.array(I_x)/Beta
            
            for k in range(len(Beta)):
                y = I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k]))
                omega_i.append(y)
            O = np.array(np.sum((omega_i)))
            return O

        E_Vyz = []
        for k in range(len(AdvIsoDF.index)-1):
            E_Vyz.append(minimize_scalar(adv_omega,bounds=bounds,args=(k), method = 'bounded').x)
        self.E_Vyz = np.array(E_Vyz)

    def get_EVyz(self):
        """
        Getter for E_Vyz
        """
        return self.E_Vyz

