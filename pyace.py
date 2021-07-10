#!/usr/bin/python3.7
import numpy as np
import pandas as pd
from   scipy.interpolate import interp1d
from   scipy.optimize    import minimize_scalar
import scipy.special     as     sp
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import t

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
        self.DFlis          = []              #list of DataFrames containing data
        self.Beta           = []              #list of heating rates
        self.BetaCC         = []              #list of correlation coefficient for T vs t
        self.files          = []              #list of files containing raw data
        self.da_dt          = []              #list of experimental conversion rates 
        self.T              = []              #list of experimental temperature
        self.t              = []              #list off experimental time
        self.TempIsoDF      = pd.DataFrame()  #Isoconversional temperature DataFrame
        self.timeIsoDF      = pd.DataFrame()  #Isoconversional time DataFrame
        self.diffIsoDF      = pd.DataFrame()  #Isoconversional conversion rate DataFrame
        self.TempAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional temperature DataFrame
        self.timeAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional time DataFrame
        self.alpha          = []              #list of experimental conversion
        self.d_a            = 0.00001               #value of alpha step for aVy method
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
    def data_extraction(self,encoding='utf8'):
        """
        Method to extract the data contained in the files
        into a list of DataFrames. Adds three columns: one
        corresponding to the Temperature in Kelvin and
        other corresponding to the conversion ('alpha')
        and a third for d(alpha)/dt.
        Also computes The heating rate ('Beta') with 
        its Correlation Coefficient.
        """
        BetaCorrCoeff = self.BetaCC
        DFlis         = self.DFlis
        Beta          = self.Beta
        filelist      = self.files
        alpha         = self.alpha
        da_dt         = self.da_dt
        T             = self.T
        t             = self.t        

        for item in filelist:

            try: 
                DF = pd.read_table(item, sep = '\t', encoding = encoding)
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15
                DF[r'$\alpha$'] = (DF[DF.columns[2]][0]-DF[DF.columns[2]])/(DF[DF.columns[2]][0]-DF[DF.columns[2]][DF.shape[0]-1])
                dadt = []
                for r in range(len(DF.index)-1):
                    try:
                        dadt.append(DF[r'$\alpha$'][r+1]-DF[r'$\alpha$'][r])/(DF[DF.columns[0]][r+1]-DF[DF.columns[0]][r])
                    except TypeError:
                        pass
                DF[r'$d\alpha/dt$'] = DF[DF.columns[0]]
                DF[r'$d\alpha/dt$'][0] = np.nan
                DF[r'$d\alpha/dt$'][1:] = dadt  

            except IndexError: 
                DF = pd.read_table(item, sep = ',', encoding = encoding)
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15
                DF[r'$\alpha$'] = (DF[DF.columns[2]][0]-DF[DF.columns[2]])/(DF[DF.columns[2]][0]-DF[DF.columns[2]][DF.shape[0]-1])
                dadt = []
                for r in range(len(DF.index)-1):
                    try:
                        dadt.append(DF[r'$\alpha$'][r+1]-DF[r'$\alpha$'][r])/(DF[DF.columns[0]][r+1]-DF[DF.columns[0]][r])
                    except TypeError:
                        pass
                DF[r'$d\alpha/dt$'] = DF[DF.columns[0]]
                DF[r'$d\alpha/dt$'][0] = np.nan
                DF[r'$d\alpha/dt$'][1:] = dadt  

            LR = linregress(DF[DF.columns[0]],DF[DF.columns[3]])

            BetaCorrCoeff.append(LR.rvalue)
            Beta.append(LR.slope)
            DFlis.append(DF)           

        for i in range(len(DFlis)):
            a = [DFlis[i][r'$\alpha$'].values[0]]
            Temp = [DFlis[i]['Temperature [K]'].values[0]]
            time = [DFlis[i][DFlis[i].columns[0]].values[0]]
            diff = [DFlis[i][r'$d\alpha/dt$'].values[1]] 
            for j in range(len(DFlis[i][r'$\alpha$'].values)):
                if DFlis[i][r'$\alpha$'].values[j] == a[-1]:
                    pass
                elif DFlis[i][r'$\alpha$'].values[j] > a[-1]:
                    a.append(DFlis[i][r'$\alpha$'].values[j])
                    Temp.append(DFlis[i]['Temperature [K]'].values[j])
                    time.append(DFlis[i][DFlis[i].columns[0]].values[j])
                    diff.append(DFlis[i][r'$d\alpha/dt$'].values[j])
                else:
                    pass
            alpha.append(np.array(a))
            T.append(np.array(Temp))
            t.append(np.array(time))
            da_dt.append(np.array(diff))

        self.BetaCC = BetaCorrCoeff
        self.DFlis  = DFlis
        self.Beta   = Beta
        self.da_dt  = da_dt
        self.T      = T
        self.t      = t
        self.alpha  = alpha
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
    	which corresponds to te fila with least 
	    data points.
        """
        alpha       = self.alpha
        da_dt       = self.da_dt
        T           = self.T
        t           = self.t
        DFlis       = self.DFlis
        TempIsoDF   = pd.DataFrame() 
        timeIsoDF   = pd.DataFrame()  
        diffIsoDF   = pd.DataFrame()
        Beta        = self.Beta

        alps = np.array(alpha[-1])

        TempIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(T[-1], decimals = 4)
        timeIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(t[-1], decimals = 4)        
        diffIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(da_dt[-1], decimals = 4)        
        
        for i in range(len(Beta)-1):
            inter_func = interp1d(alpha[i],
                                  t[i], 
                                  kind='cubic', 
                                  bounds_error=False, 
                                  fill_value="extrapolate")
            timeIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)
            
            inter_func2 = interp1d(alpha[i], 
                                   T[i], 
                                   kind='cubic', 
                                   bounds_error=False, 
                                   fill_value="extrapolate")
            TempIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func2(alps), decimals = 4)
            
            inter_func3 = interp1d(alpha[i], 
                                   da_dt[i], 
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


        self.TempIsoDF    = TempIsoDF 	
        self.timeIsoDF    = timeIsoDF
        self.diffIsoDF    = diffIsoDF
#-----------------------------------------------------------------------------------------------------------
    def get_df_isoconv(self):
        """
        Getter dataframe isoconversional. 
        """
        return self.TempIsoDF
#-----------------------------------------------------------------------------------------------------------
    def adv_isoconversional(self, method='points', N = 1000, d_a = 0.001):
        """
        Method that builds a DataFrame
		based on the advanced Vyazovkin 
		method. The index correspond to 
		an equidistant set of data from 
		the first calculated value in 
		the first element of DFlis.  
        """
        TempAdvIsoDF   = pd.DataFrame()
        timeAdvIsoDF   = pd.DataFrame()        
        Beta           = self.Beta
        alpha          = self.alpha
        T              = self.T
        t              = self.t    
  
        if method == 'points':
            alps, d_a = np.linspace(alpha[-1][0],alpha[-1][-1],N,retstep=True)
        elif method == 'interval':
            alps = np.arange(alpha[-1][0],alpha[-1][-1],d_a)
        else:
            raise ValueError('Method not recognized')

        for i in range(0,len(Beta)):
            inter_func = interp1d(alpha[i], 
                                  T[i],
                                  kind='cubic', bounds_error=False, 
                                  fill_value="extrapolate")
            TempAdvIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func(alps), decimals = 4)

            inter_func2 = interp1d(alpha[i], 
                                   t[i],
                                   kind='cubic', bounds_error=False, 
                                   fill_value="extrapolate")
            timeAdvIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func2(alps), decimals = 4)
        timeAdvIsoDF.index = alps
        TempAdvIsoDF.index = alps

        self.d_a          = d_a
        self.TempAdvIsoDF = TempAdvIsoDF
        self.timeAdvIsoDF = timeAdvIsoDF        
#-----------------------------------------------------------------------------------------------------------        
    def get_adviso(self):
        """
        Getter for dataframe AdvIso
        """
        return self.TempAdvIsoDF

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
    def save_df(self, E_Fr= None, E_OFW=None, E_KAS=None, E_Vy=None, E_aVy=None, dialect="xlsx" ):
        """
        Method to save dataframe with
        values calculated by ActivationEnergy
        """
        TempIsoDF  = self.TempIsoDF
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

        alps = TempIsoDF.index.values

        columns = ['alpha']
        if np.any(E_OFW)!=None:
            columns.append('OFW')
        if np.any(E_KAS)!=None:
            columns.append('KAS')
        if np.any(E_Vy)!=None:
            columns.append('Vyazovkin')
        if np.any(E_aVy)!=None:
            columns.append('adv.Vyazovkin')
        DF_nrgy = pd.DataFrame([], columns = columns)
        DF_nrgy['alpha']  = alps
        if 'OFW' in columns:
            DF_nrgy['OFW']=E_OFW
        if 'KAS' in columns:
            DF_nrgy['KAS'] = E_KAS
        if 'Vyazovkin' in columns:
            DF_nrgy['Vyazovkin'] = E_Vy
        if 'adv.Vyazovkin' in columns:
            DF_nrgy['adv.Vyazovkin'][0]  = np.nan
            DF_nrgy['adv.Vyazovkin'][1:] = E_aVy

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
                nombre = 'HR={0:0.3}_K_per_min.csv'.format(Beta[i])    
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
            raise ValueError("Dialect not recognized")
#-----------------------------------------------------------------------------------------------------------
    def get_avsT_plot(self)
        for i in range(len(self.DFlis)):
            plt.plot(self.T[i],
                     self.alpha[i],
                     label=str(np.round(self.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[3])
            plt.ylabel(self.DFlis[i].columns[4])
            plt.axis('equal')
            plt.legend(loc='lower right')
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def get_dadtvsT_plot(self)
        for i in range(len(self.DFlis)):
            plt.plot(self.T[i],
                     self.da_dt[i],
                     label=str(np.round(self.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[3])
            plt.ylabel(self.DFlis[i].columns[5])
            plt.axis('equal')
            plt.legend(loc='lower right')
        return plt.show()

#-----------------------------------------------------------------------------------------------------------
    def get_avst_plot(self)
        for i in range(len(self.DFlis)):
            plt.plot(self.t[i],
                     self.alpha[i],
                     label=str(np.round(sel.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[0])
            plt.ylabel(self.DFlis[i].columns[4])
            plt.axis('equal')
            plt.legend(loc='lower right')
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def get_dadtvst_plot(self)
        for i in range(len(self.DFlis)):
            plt.plot(self.t[i],
                     self.da_dt[i],
                     label=str(np.round(self.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[0])
            plt.ylabel(self.DFlis[i].columns[5])
            plt.axis('equal')
            plt.legend(loc='lower right')
        return plt.show()
#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
class ActivationEnergy(DataExtraction):
    """
	Class that uses the lists and
	Data Frames generated with 
	Dataextraction to compute 
	Activation energy values based on 
	four methods: FOW, KAS, Vyazovkin
	and Advanced Vyazovkin. 
    """
    def __init__(self):
        """
		Constructor. Receives the Isoconversional
		Data Frame as first parameter, the list 'Beta'
		as second parameter, and the Adv_Isoconversional
		Data Frame as an optional third parameter.
        """
        self.E_OFW = []
        self.E_KAS = []
        self.E_Vy  = []
        self.E_aVy = []
        self.IsoDF = self.TempIsoDF
        self.Beta  = Beta
        self.logB  = np.log(Beta)

        if(adv_df is not None):
            self.AdvIsoDF = adv_df
        """ 
        Universal gas constant
        0.0083144626 kJ/(mol*K)
        """
        self.R    = 0.0083144626
        self.tinv = lambda p, df: abs(t.ppf(p/2, df))
        self.ts   = tinv(0.05, len(x)-2)
#-----------------------------------------------------------------------------------------------------------
    def Fr(self):
        """
        Method to compute the Activation 
	    Energy based on the Osawa-Flynn-Wall 
	    (OFW) treatment.
        """
        E_Fr      = []
        E_Fr_err  = []
        IsoDF     = self.IsoDF
        diffIsoDF = self.diffIsoDF
        Fr_b      = []

        for i in range(0,IsoDF.shape[0]):  
            y = diffIsoDF.iloc[i].values
            x = 1/(IsoDF.iloc[i].values)
            LR = linregress(x,y)
            E_a_i = -(self.R/1.052)*(LR.slope)
            error = -(self.R/1.052)*(LR.stderr)
            Fr_b.append(LR.intercept)
            E_Fr_err.append(ts*error)
            E_Fr.append(E_a_i)

        self.E_Fr   = np.array(E_Fr)
        self.Fr_95e = np.array(E_Fr_err)
        self.Fr_b   = np.array(Fr_b) 

        return self.E_Fr, self.Fr_95e, self.Fr_b

#-----------------------------------------------------------------------------------------------------------
    def OFW(self):
        """
        Method to compute the Activation 
	    Energy based on the Osawa-Flynn-Wall 
	    (OFW) treatment.
        """
        logB       = self.logB
        E_OFW      = []
        E_OFW_err  = []
        IsoDF      = self.IsoDF

        for i in range(0,IsoDF.shape[0]):  
            y = (logB)
            x = 1/(IsoDF.iloc[i].values)
            LR = linregress(x,y)
            E_a_i = -(self.R/1.052)*(LR.slope)
            error = -(self.R/1.052)*(LR.stderr)
            E_OFW_err.append(ts*error)
            E_OFW.append(E_a_i)

        self.E_OFW   = np.array(E_OFW)
        self.OFW_95e = np.array(E_OFW_err)

        return self.E_OFW, self.OFW_95e
#-----------------------------------------------------------------------------------------------------------
    def KAS(self):
        """
        Method to compute the Activation 
		Energy based on the 
		Kissinger-Akahira-Sunose (KAS) treatment.
        """

        logB       = self.logB
        IsoDF      = self.IsoDF
        E_KAS      = []
        E_KAS_err  = []

        for i in range(0,IsoDF.shape[0]):     
            y = (logB)- np.log((IsoDF.iloc[i].values)**1.92)
            x = 1/(IsoDF.iloc[i].values)
            LR = linregress(x,y)
            E_a_i = -(self.R)*(LR.slope)
            error = -(self.R)*(LR.stderr)
            E_KAS_err.append(ts*error)
            E_KAS.append(E_a_i)

        self.E_KAS   = np.array(E_KAS)
        self.KAS_95e = np.array(E_KAS_err)
        return self.E_KAS, self.KAS_95e
#-----------------------------------------------------------------------------------------------------------
    def omega(self,E,row,DF,Beta,method = 'senum-yang'):
    
        T0      = DF.iloc[0].values   
        T       = DF.iloc[row].values

        def senum_yang(E):
            x = E/(self.R*T)
            num = x**3 + 18*(x**2) + 88*x + 96
            den = x**4 + 20*(x**3) + 120*(x**2) +240*x +120
            s_y = ((np.exp(-x))/x)*(num/den)
            return (E/self.R)*s_y
    
        def trapezoid(E):
            x0     = T0
            y0     = np.exp(-E/(self.R*x0))
            xf     = T
            yf     = np.exp(-E/(self.R*xf))
            tpz    = []
            for i in range(len(T)):
                tpz.append(integrate.trapezoid([y0[i],yf[i]],
                                               [x0[i],xf[i]]))
            return np.array(tpz)
    
        def quad(E):
        
            def integral(x,E):
                return np.exp(-E/(self.R*x))
            
            quad    = []
            for i in range(len(T)):
                quad.append(integrate.quad(integral,
                                           T0[i],
                                           T[i],
                                           args=(E))[0])
            return np.array(quad)
          
        omega_i = []

        if method == 'senum-yang':
            p = senum_yang(E)
            p_B = p/Beta
            for j in range(len(Beta)):
                y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
                omega_i.append(y)
            return np.array(np.sum((omega_i)))

        elif method == 'trapezoid':
            p = trapezoid(E)
            p_B = p/Beta
            for j in range(len(Beta)):
                y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
                omega_i.append(y)
            return np.array(np.sum((omega_i)))

        elif method == 'quad':
            p = quad(E)
            p_B = p/Beta
            for j in range(len(Beta)):
                y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
                omega_i.append(y)
            return np.array(np.sum((omega_i)))

#-----------------------------------------------------------------------------------------------------------
    def set_bounds(self, bounds):
        """
        Setter for bounds variable
        """
        self.bounds = bounds
        print("The bounds for evaluating E are "+str(bounds))
        return bounds
#-----------------------------------------------------------------------------------------------------------
    def visualize_omega(self,row,bounds=(1,300),N=1000,method = 'senum-yang'):
        """
        Method to visualize omega function.
        Bounds requiered from function vy o 
        by bounds setter
        """
        
        E = np.linspace(bounds[0], bounds[1], N)
        O = np.array([float(self.omega(E[i],row,self.IsoDF,self.Beta,method = method)) for i in range(len(E))])
        plt.style.use('seaborn')
        plt.plot(E,O,color='teal',label=r'$\alpha$ = '+str(np.round(self.IsoDF.index[row],decimals=3)))        
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
        plt.legend()

        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def Vy(self, bounds,method='senum-yang'):
        """
        Method to compute the Activation 
		Energy based on the Vyazovkin treatment.
        """
        E_Vy       = []
        Tempdf     = self.IsoDF
        Beta       = self.Beta 
        
        for k in range(len(Tempdf.index)):
            E_Vy.append(minimize_scalar(self.omega, args=(k,Tempdf,Beta,method),bounds=bounds, method = 'bounded').x)

        self.E_Vy = np.array(E_Vy)
        return self.E_Vy
#-----------------------------------------------------------------------------------------------------------        
    def J(self, E, inf, sup):
        """
        Temperature integral for the
        Advanced Vyazovkin Treatment
        """
        
        a=E/(self.R)
        b=inf
        c=sup

        return a*(sp.expi(-a/c)-sp.expi(-a/b)) + c*np.exp(-a/c) - b*np.exp(-a/b)
#-----------------------------------------------------------------------------------------------------------        
    def adv_omega(self,E,row):
        """
        Function to minimize according
        to the advanced Vyazovkin treatment
        """
        AdvIsoDF = self.AdvIsoDF
        Beta     = self.Beta
        j = row
    
        I_x = np.array([self.J(E,
                      AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j]],
                      AdvIsoDF[AdvIsoDF.columns[i]][AdvIsoDF.index[j+1]]) 
                      for i in range(len(AdvIsoDF.columns))])
        I_B = I_x/Beta
        
        omega_i = np.array([I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k])) for k in range(len(Beta))])
 
        O = np.array(np.sum((omega_i)))
        return O
#-----------------------------------------------------------------------------------------------------------
    def visualize_advomega(self,row,bounds=(1,300),N=1000):
        """
        Method to visualize adv_omega function.
        Bounds requiered from function Vy o 
        by bounds setter
        """
        
        E = np.linspace(bounds[0], bounds[1], N)
        O = np.array([float(self.adv_omega(E[i],row)) for i in range(len(E))])
        plt.style.use('seaborn')
        plt.plot(E,O,color='teal',label=r'$\alpha$ = '+str(np.round(self.IsoDF.index[row],decimals=3)))
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
        plt.legend()
    
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def aVy(self,bounds):
        """
        Method to compute the Activation 
		Energy based on the Advanced
		Vyazovkin treatment.
        """

        AdvIsoDF = self.AdvIsoDF 
        Beta     = self.Beta

        E_aVy = []
        for k in range(len(AdvIsoDF.index)-1):
            E_aVy.append(minimize_scalar(self.adv_omega,bounds=bounds,args=(k), method = 'bounded').x)
        self.E_aVy = np.array(E_aVy)
        return self.E_aVy
#-----------------------------------------------------------------------------------------------------------        
    def prediction(self, E = None, B = 1, T0 = 298.15, Tf=1298.15):
        """
        Method to calculate a kinetic
        prediction, based on an
        isoconversional activation energy
        """
        b      = self.Fr_b
        a_pred = [0]
        T      = np.linspace(T0,Tf,len(b))
        t      =  (T-T0)/B
        for i in range(len(self.Fr_b)):
            a = a[i] + b[i]*np.exp(-(E[i]/(self.R*T[i]))*t[i])
            a_pred.append(a)

        a_pred      = np.array(a_pred)
        self.a_pred = a_pred
       
        return self.a_pred
#-----------------------------------------------------------------------------------------------------------
    def get_Eavsa_plot(self, E_Fr= None, E_OFW=None, E_KAS=None, E_Vy=None, E_aVy=None)
        
        plt.plot(self.diffIsoDF.index.values,
                Fr,
                label='Fr')
        plt.plot(self.TempIsoDF.index.values,
                OFW,
                label='OFW')  
        plt.plot(self.TempIsoDF.index.values,
                KAS,
                label='KAS')    
        plt.plot(self.TempIsoDF.index.values,
                Vy,
                label='Vyazovkin')    
        plt.plot(self.TempAdvIsoDF.index.values[1:],
                aVy,
                label='adv. Vyazovkin')   
        ax.set_ylabel(r'$E_{\alpha}$')
        ax.set_xlabel(r'$\alpha$')
        ax.legend()

