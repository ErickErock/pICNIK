#!/usr/bin/python3.7
import numpy as np
import pandas as pd
from   scipy.interpolate import interp1d
from   scipy.optimize    import minimize_scalar
import scipy.special     as     sp
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------
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
        self.DFlis          = []  #ToDo:Change Dflis name for another more thermochemical
        self.Beta           = []
        self.BetaCC         = []
        self.files          = []
        self.da_dt          = []
        self.T              = []
        self.t              = []
        self.TempIsoDF      = pd.DataFrame()
        self.timeIsoDF      = pd.DataFrame() 
        self.diffIsoDF      = pd.DataFrame() 
        self.TempAdvIsoDF   = pd.DataFrame()
        self.timeAdvIsoDF   = pd.DataFrame()
        self.alpha          = []
#-----------------------------------------------------------------------------------------------------------    
    def set_data(self, filelist):
        """
        Method to establish the file list
        for the extrator. The list must be 
        in ascendent heating rate order.
        """
        self.files = filelist
        print("Files to be used: \n{} ".format(filelist))
#-----------------------------------------------------------------------------------------------------------        
    def data_extraction(self):
        """
        Method to extract the data contained in the files
        into a list of DataFrames. Adds two columns: one
        corresponding to the Temperature in Kelvin and
        other corresponding to the conversion ('alpha').
        Also computes The heating rate ('Beta') with 
        its Pearson Coefficient.
        """
        BetaCorrCoeff = self.BetaCC
        DFlis          = self.DFlis
        Beta           = self.Beta
        filelist       = self.files
        

        for item in filelist:

            #ToDo: probar con utf8
            try: 
                DF = pd.read_table(item, sep = '\t', encoding = 'latin_1')
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15
                DF[r'$\alpha$'] = (DF[DF.columns[2]][0]-DF[DF.columns[2]])/(DF[DF.columns[2]][0]-DF[DF.columns[2]][DF.shape[0]-1])
                da_dt = []
                for r in range(len(DF.index)-1):
                    try:
                        da_dt.append(DF[r'$\alpha$'][r+1]-DF[r'$\alpha$'][r])/(DF['Time (min)'][r+1]-DF['Time (min)'][r])
                    except TypeError:
                        pass
                DF[r'$d\alpha/dt$'] = DF['Time (min)']
                DF[r'$d\alpha/dt$'][0] = np.nan
                DF[r'$d\alpha/dt$'][1:] = da_dt  

            except IndexError: 
                DF = pd.read_table(item, sep = ',', encoding = 'latin_1')
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15
                DF[r'$\alpha$'] = (DF[DF.columns[2]][0]-DF[DF.columns[2]])/(DF[DF.columns[2]][0]-DF[DF.columns[2]][DF.shape[0]-1])
                da_dt = []
                for r in range(len(DF.index)-1):
                    try:
                        da_dt.append(DF[r'$\alpha$'][r+1]-DF[r'$\alpha$'][r])/(DF['Time (min)'][r+1]-DF['Time (min)'][r])
                    except TypeError:
                        pass
                DF[r'$d\alpha/dt$'] = DF['Time (min)']
                DF[r'$d\alpha/dt$'][0] = np.nan
                DF[r'$d\alpha/dt$'][1:] = da_dt  

            y = DF['Temperature [K]']
            x = DF['Time (min)']
            den = np.sum((x-(np.mean(x)))**2)
            num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
            r = (num**2)/(np.sum((x-np.mean(x))**2)*(np.sum((y-np.mean(y))**2)))

            BetaCorrCoeff.append(r)
            Beta.append(num/den )
            DFlis.append(DF)
        self.BetaCC = BetaCorrCoeff
        self.DFlis  = DFlis
        self.Beta   = Beta
#-----------------------------------------------------------------------------------------------------------        
    def get_beta(self):
        """
        Getter for Beta
        """
        return self.Beta
#-----------------------------------------------------------------------------------------------------------
    def get_betaCC(self):
        """
        Getter for the Beta
        Pearson Coefficients
        """
        return self.BetaCC
 #-----------------------------------------------------------------------------------------------------------       
    def get_DFlis(self):
        """
        Getter for the DataFrame
        list
        """
        return self.DFlis
#-----------------------------------------------------------------------------------------------------------
    def isoconversional(self):
        """
        Method that builds a DataFrame based 
        on the isoconversional principle by
		building a function interpolated from
		the data frame with least data points,
		wich corresponds to te fila with least 
		data points.
        """
        alpha       = self.alpha
        da_dt       = self.da_dt
        T           = self.T
        t           = self.t
        DFlis       = self.DFlis
        TempIsoDF   = self.TempIsoDF 
        timeIsoDF   = self.timeIsoDF  
        diffIsoDF   = self.diffIsoDF
        Beta        = self.Beta

        for i in range(len(DFlis)):
            a = [DFlis[i][r'$\alpha$'].values[0]]
            Temp = [DFlis[i]['Temperature [K]'].values[0]]
            time = [DFlis[i]['Time (min)'].values[0]]
            diff = [DFlis[i][r'$d\alpha/dt$'].values[1]] 
            for j in range(len(DFlis[i][r'$\alpha$'].values)):
                if DFlis[i][r'$\alpha$'].values[j] == a[-1]:
                    pass
                elif DFlis[i][r'$\alpha$'].values[j] > a[-1]:
                    a.append(DFlis[i][r'$\alpha$'].values[j])
                    Temp.append(DFlis[i]['Temperature [K]'].values[j])
                    time.append(DFlis[i]['Time (min)'].values[j])
                    diff.append(DFlis[i][r'$d\alpha/dt$'].values[j])
                else:
                    pass
            alpha.append(a)
            T.append(Temp)
            t.append(time)
            da_dt.append(diff)

        alps = np.array(alpha[-1])

        TempIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(np.array(T[-1]), decimals = 4)
        timeIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(np.array(t[-1]), decimals = 4)        
        diffIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(np.array(da_dt[-1]), decimals = 4)        
        
        for i in range(len(Beta)-1):
            inter_func = interp1d(np.array(alpha[i]),
                                  np.array(t[i]), 
                                  kind='cubic', 
                                  bounds_error=False, 
                                  fill_value="extrapolate")
            timeIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)
            
            inter_func2 = interp1d(np.array(alpha[i]), 
                                   np.array(T[i]), 
                                   kind='cubic', 
                                   bounds_error=False, 
                                   fill_value="extrapolate")
            TempIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func2(alps), decimals = 4)
            
            inter_func3 = interp1d(np.array(alpha[i]), 
                                  np.array(da_dt[i]), 
                                  kind='cubic', 
                                  bounds_error=False, 
                                  fill_value="extrapolate")
            diffIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func3(alps), decimals = 4)
          
        colnames          = TempIsoDF.columns.tolist()
        colnames          = colnames[1:] + colnames[:1]

        TempIsoDF.index   = alpha[-1]
        TempIsoDF         = TempIsoDF[colnames]
        timeIsoDF.index   = alpha[-1]
        timeIsoDF         = timeIsoDF[colnames]
        diffIsoDF.index   = alpha[-1]
        diffIsoDF         = diffIsoDF[colnames]


        self.da_dt   	  = da_dt
        self.T      	  = T
        self.t      	  = t
        self.TempIsoDF    = TempIsoDF 	
        self.timeIsoDF    = timeIsoDF
        self.diffIsoDF    = diffIsoDF
#-----------------------------------------------------------------------------------------------------------
    def get_df_isoconv(self):
        """
        Getter dataframe isoconversional. 
        """
        return self.Iso_convDF
#-----------------------------------------------------------------------------------------------------------
    def adv_isoconversional(self):
        """
        Method that builds a DataFrame
		based on the advanced Vyazovkin 
		method. The index correspond to 
		an equidistant set of data from 
		the first calculated value in 
		the first element of DFlis.  
        """
        TempAdvIsoDF   = self.TempAdvIsoDF
        timeAdvIsoDF   = self.timeAdvIsoDF        
        Beta       = self.Beta
        alpha      = self.alpha
        T          = self.T
        t          = self.t
        
        alps = np.array(alpha[-1])
        for i in range(0,len(Beta)):
            inter_func = interp1d(alpha[i], 
                                  T[i],
                                  kind='cubic', bounds_error=False, 
                                  fill_value="extrapolate")
            TempAdvIsoDF['HR '+str(np.round(Beta[i], decimals = 0)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)

            inter_func2 = interp1d(alpha[i], 
                                  t[i],
                                  kind='cubic', bounds_error=False, 
                                  fill_value="extrapolate")
            timeAdvIsoDF['HR '+str(np.round(Beta[i], decimals = 0)) + ' K/min'] = np.round(inter_func2(alps), decimals = 4)
        timeAdvIsoDF.index = alps
        TempAdvIsoDF.index = alps

        self.TempAdvIsoDF = TempAdvIsoDF
        self.timeAdvIsoDF = timeAdvIsoDF        
#-----------------------------------------------------------------------------------------------------------        
    def get_adviso(self):
        """
        Getter for dataframe AdvIso
        """
        return self.AdvIsoDF

#-----------------------------------------------------------------------------------------------------------
    def get_dadt(self):
        """
        Getter for da_dt
        """
        return self.da_dt
#-----------------------------------------------------------------------------------------------------------
    def get_t(self):
        """
        Getter for t
        """
        return self.t
#-----------------------------------------------------------------------------------------------------------    
    def get_T(self):
        """
        Getter for T
        """
        return self.T
#-----------------------------------------------------------------------------------------------------------
    def get_values(self):
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
#-----------------------------------------------------------------------------------------------------------
    def save_df(self, E_FOW, E_KAS, E_vy, E_Vyz, dialect="xlsx" ):
        """
        Method to save dataframe with
        values calculated by ActivationEnergy
        """
        IsoDF      = self.IsoDF
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

        alps = IsoDF.index.values

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

#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
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
        self.IsoDF = iso_df
        self.Beta  = Beta
        self.logB = np.log(Beta)
        self.alpha = np.linspace(np.array(iso_df.index)[0], 
                                 np.array(iso_df.index)[-1],18000)
        if(adv_df is not None):
            self.AdvIsoDF = adv_df
        """ 
        Universal gas constant
        0.0083144626 kJ/(mol*K)
        """
        self.R =0.0083144626
#-----------------------------------------------------------------------------------------------------------
    def FOW(self):
        """
        Method to compute the Activation 
	    Energy based on the Flynn-Osawa-Wall 
	    (FOW) treatment.
        """
        logB       = self.logB
        E_FOW      = self.E_FOW
        IsoDF      = self.IsoDF
        for i in range(0,IsoDF.shape[0]):  
            y = (logB)
            x = 1/(IsoDF.iloc[i].values)
            den = np.sum((x-(np.mean(x)))**2)
            num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
            E_a_i = -(self.R/1.052)*(num/den)
            E_FOW.append(E_a_i)
        self.E_FOW = np.array(E_FOW)
        return self.E_FOW
#-----------------------------------------------------------------------------------------------------------
    def KAS(self):
        """
        Method to compute the Activation 
		Energy based on the 
		Kissinger-Akahira-Sunose (KAS) treatment.
        """

        logB       = self.logB
        IsoDF      = self.IsoDF
        E_KAS      = self.E_KAS
        for i in range(0,IsoDF.shape[0]):     
            y = (logB)- np.log((IsoDF.iloc[i].values)**1.92)
            x = 1/(IsoDF.iloc[i].values)
            den = np.sum((x-(np.mean(x)))**2)
            num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
            E_a_i = -(self.R)*(num/den )
            E_KAS.append(E_a_i)
        self.E_KAS = np.array(E_KAS)
        return self.E_KAS
#-----------------------------------------------------------------------------------------------------------
    def omega(self, E, row):
        """
        Function to minimize according
        to the Vyazovkin treatment.
        """
        Tempdf     = self.IsoDF
        Beta       = self.Beta

        x = E/(self.R*Tempdf.iloc[row])
        omega_i = []
        px = ((np.exp(-x))/x)*(((x**3)+(18*(x**2))+(88*x)+96)/((x**4)+(20*(x**3))+(120*(x**2))+(240*x)+120))
        p_B = px/Beta
        for j in range(len(Beta)):
            y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
            omega_i.append(y)
        O = np.array(np.sum((omega_i)))
        
        return O
#-----------------------------------------------------------------------------------------------------------
    def set_bounds(self, bounds):
        """
        Setter for bounds variable
        """
        self.bounds = bounds
        return bounds
#-----------------------------------------------------------------------------------------------------------
    def visualize_omega(self,row,bounds=(1,300),N=1000):
        """
        Method to visualize omega function.
        Bounds requiered from function vy o 
        by bounds setter
        """
        
        E = np.linspace(bounds[0], bounds[1], N)
        O = np.array([float(self.omega(E[i],row)) for i in range(len(E))])
        
        plt.plot(E,O)
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')

        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def vy(self, bounds):
        """
        Method to compute the Activation 
		Energy based on the Vyazovkin treatment.
        """
        E_vy       = self.E_vy
        Tempdf     = self.IsoDF
        Beta       = self.Beta 
        for k in range(len(Tempdf.index)):
            E_vy.append(minimize_scalar(self.omega, args=(k),bounds=bounds, method = 'bounded').x)

        self.E_vy = np.array(E_vy)
        return self.E_vy
#-----------------------------------------------------------------------------------------------------------        
    def I(self, E, inf, up):
        """
        Temperature integral for the
        Advanced Vyazovkin Treatment
        """
        
        a=E/(self.R)
        b=inf
        c=up

        return a*(sp.expi(-a/c)-sp.expi(-a/b)) + c*np.exp(-a/c) - b*np.exp(-a/b)
#-----------------------------------------------------------------------------------------------------------        
    def adv_omega(self,E,rowi):
        """
        Function to minimize according
        to the advanced Vyazovkin treatment
        """
        AdvIsoDF = self.AdvIsoDF
        Beta     = self.Beta
        j = rowi
        omega_i = []
        I_x = []
        for i in range(len(AdvIsoDF.columns)):
            I_x.append(self.I(E,
                         AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j]],
                         AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j+1]]))
        I_B = np.array(I_x)/Beta
        
        for k in range(len(Beta)):
            y = I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k]))
            omega_i.append(y)
        O = np.array(np.sum((omega_i)))
        return O
#-----------------------------------------------------------------------------------------------------------
    def visualize_advomega(self,row,bounds=(1,300),N=1000):
        """
        Method to visualize adv_omega function.
        Bounds requiered from function vy o 
        by bounds setter
        """
        
        E = np.linspace(bounds[0], bounds[1], N)
        O = np.array([float(self.adv_omega(E[i],row)) for i in range(len(E))])
        
        plt.plot(E,O)
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
    
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def vyz(self,bounds):
        """
        Method to compute the Activation 
		Energy based on the Advanced
		Vyazovkin treatment.
        """

        AdvIsoDF = self.Adv_IsoDF 
        Beta     = self.Beta

        E_Vyz = []
        for k in range(len(AdvIsoDF.index)-1):
            E_Vyz.append(minimize_scalar(adv_omega,bounds=bounds,args=(k), method = 'bounded').x)
        self.E_Vyz = np.array(E_Vyz)
        return self.E_Vyz
#-----------------------------------------------------------------------------------------------------------        
    def DeltaAlpha(self):

        return np.round(self.AdvIsoDF.index.values[1] -
                        self.AdvIsoDF.index.values[0], decimals=5)


