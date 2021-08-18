#!/usr/bin/python3.7

#Dependencies
import numpy as np
import pandas as pd
from   scipy.interpolate import interp1d
from   scipy.optimize    import minimize_scalar
import scipy.special     as     sp
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import t
from scipy import integrate
import derivative
#-----------------------------------------------------------------------------------------------------------
class DataExtraction(object):
    """
    Class that manipulates raw data to create lists and Data Frames 
    that will be used to compute the Activation Energy.
    """
    def __init__(self):
        """
        Constructor. 

        Parameters:    None

        Notes:         It only defines variables.
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
        self.d_a            = 0.00001         #default value of alpha step for aVy method
#-----------------------------------------------------------------------------------------------------------    
    def set_data(self, filelist):
        """
        Method to establish the file list for the extrator. 

        Parameters:   filelist : list object containing the paths
                                 of the files to be used.

        Notes:        The paths must be sorted in ascendent heating 
                      rate order.
        """
        print("Files to be used: \n{} ".format(filelist))
        self.files = filelist
#-----------------------------------------------------------------------------------------------------------        
    def data_extraction(self,encoding='utf8'):
        """
        Method to extract the data contained in the files into a list of DataFrames. 
        Adds three columns: one corresponding to the absolute temperature, another 
        corresponding to the conversion ('alpha') and a third for d(alpha)/dt.
        Also computes The heating rate ('Beta') with its Correlation Coefficient.

        Parameters:   encoding: The available encodings for pandas.read_csv() method. Includes but not limited 
                                to 'utf8', 'utf16','latin1'. For more information on the python standar encoding:
                                (https://docs.python.org/3/library/codecs.html#standard-encodings)
        """
        BetaCorrCoeff = self.BetaCC
        DFlis         = self.DFlis
        Beta          = self.Beta
        filelist      = self.files
        alpha         = self.alpha
        da_dt         = self.da_dt
        T             = self.T
        t             = self.t        

        # Read the data from each csv
        # Create the Dataframe of each experiment
        # Add three columns (T,alpha,(da/dt))
        # Compute the linear regression of T vs t
        for item in filelist:

            try: 
                DF = pd.read_table(item, sep = '\t', encoding = encoding)
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15
                DF[r'$\alpha$'] = (DF[DF.columns[2]][0]-DF[DF.columns[2]])/(DF[DF.columns[2]][0]-DF[DF.columns[2]][DF.shape[0]-1])
                dadt = derivative.dxdt(DF[r'$\alpha$'],DF[DF.columns[0]],kind='spline',s=0.01,order=5)
                DF[r'$d\alpha/dt$'] = DF[DF.columns[0]]
                DF[r'$d\alpha/dt$'] = dadt  

            except IndexError: 
                DF = pd.read_table(item, sep = ',', encoding = encoding)
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15
                DF[r'$\alpha$'] = (DF[DF.columns[2]][0]-DF[DF.columns[2]])/(DF[DF.columns[2]][0]-DF[DF.columns[2]][DF.shape[0]-1])
                dadt = derivative.dxdt(DF[r'$\alpha$'],DF[DF.columns[0]],kind='spline',s=0.01,order=5)
                DF[r'$d\alpha/dt$'] = DF[DF.columns[0]]
                DF[r'$d\alpha/dt$'] = dadt   

            LR = linregress(DF[DF.columns[0]],DF[DF.columns[3]])

            BetaCorrCoeff.append(LR.rvalue)
            Beta.append(LR.slope)
            DFlis.append(DF)           

        # Create an array of sorted in ascendent order values of conversion (alpha) and arrays
        # for the temperature, time and rate of conversion corresponding to said conversion values
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
        Parameters:   None

        Returns:      list object containing the experimental heating rate sorted 
                      in ascendent order obtained from a linear regression of T vs t.
        """
        return self.Beta
#-----------------------------------------------------------------------------------------------------------
    def get_betaCC(self):
        """
        Parameters:   None

        Returns:      list object containing the experimental T vs t correlation coefficient
                      obtained from a linear regression, sorted in correspondance with the 
                      heating rate list (attribute Beta).
        """
        return self.BetaCC
#-----------------------------------------------------------------------------------------------------------       
    def get_DFlis(self):
        """
        Parameters:   None

        Returns:      list object containing the DataFrames with the experimental data, sorted 
                      in correspondance with the heating rate list (attribute Beta).
        """
        return self.DFlis
#-----------------------------------------------------------------------------------------------------------
    def isoconversional(self):
        """
        Isoconversional DataFrames building method for the Friedman, KAS, OFW and Vyazovkin methods.
        The isoconversional values for T, t and da/dt are obtained by interpolation. 

        Parameters:    None

        Returns:       None

        Notes:         This method asigns values to the attributes: TempIsoDF, timeIsoDF and diffIsoDF  
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


        # Take the experimental data set with the less data points (alps), so the interpolation is made with the
        # data sets with more experimental information.
        # Create the interpolation functions and evaluate them over the conversion values of the latter set (alps).
        # Create the isoconversional DataFrames with the conversion values (alps) as index and the 
        # interpolation values as columns corresponding to their experimental heating rates.
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
    def get_TempIsoDF(self):
        """
        Parameters:   None

        Returns:      DataFrame of isoconversional temperatures. The index is the set of conversion
                      values from the experiment with the less data points (which correspond to the
                      smallest heating rate). The columns are isoconversional temperatures, sorted in 
                      heating rate ascendent order from left to right.
        """
        return self.TempIsoDF
#-----------------------------------------------------------------------------------------------------------
    def get_timeIsoDF(self):
        """
        Parameters:   None

        Returns:      DataFrame of isoconversional times. The index is the set of conversion values 
                      from the experiment with the less data points (which correspond to the smallest 
                      heating rate). The columns are isoconversional times, sorted in heating rate 
                      ascendent order from left to right.
        """
        return self.timeIsoDF
#-----------------------------------------------------------------------------------------------------------
    def get_diffIsoDF(self):
        """
        Parameters:   None

        Returns:      DataFrame of isoconversional conversion rates. The index is the set of conversion 
                      values from the experiment with the less data points (which correspond to the smallest 
                      heating rate). The columns are isoconversional conversion rates, sorted in heating 
                      rate ascendent order from left to right.
        """
        return self.timeIsoDF
#-----------------------------------------------------------------------------------------------------------
    def adv_isoconversional(self, method='points', N = 1000, d_a = 0.001):
        """
        Isoconversional DataFrames building method for the advanced Vyazovkin method. The isoconversional 
        values for T and t are obtained by interpolation. 

        Parameters:     method : String. Value can be either 'points' or 'interval'. ṕoints'is the
                                 default value.

                        N      : Int. Number of conversion points if the 'points' method is given.
                                 1000 is the default value.

                        d_a    : Float. Size of the interval between conversion values if the method 
                                 'interval' is given. 0.001 is the default value.    
        
        Returns:       None

        Notes:         This method asigns values to the attributes: TempAdvIsoDF, timeAdvIsoDF and d_a  
        """

        TempAdvIsoDF   = pd.DataFrame()
        timeAdvIsoDF   = pd.DataFrame()        
        Beta           = self.Beta
        alpha          = self.alpha
        T              = self.T
        t              = self.t    

        # Evaluate which methd was given and create an array of conversion values (alps)
        # Create interpolation functions and evaluate on the conversion values (alps)
        # Create the isoconversional DataFrames with the conversion values (alps) as index and the 
        # interpolation values as columns corresponding to their experimental heating rates.
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
    def get_TempAdvIsoDF(self):
        """
        Parameters:   None

        Returns:      DataFrame of isoconversional temperatures for the advanced Vyazovkin method. 
                      The index is a set of equidistant (attribute d_a) conversion values, with 
                      initial and final points taken from the experiment with the less data points 
                      (which correspond to the smallest heating rate). The columns are isoconversional 
                      temperatures, sorted in heating rate ascendent order from left to right.
        """
        return self.TempAdvIsoDF
#-----------------------------------------------------------------------------------------------------------        
    def get_timeAdvIsoDF(self):
        """
        Parameters:   None

        Returns:      DataFrame of isoconversional times for the advanced Vyazovkin method. 
                      The index is a set of equidistant (attribute d_a) conversion values, with 
                      initial and final points taken from the experiment with the less data points 
                      (which correspond to the smallest heating rate). The columns are isoconversional 
                      times, sorted in heating rate ascendent order from left to right.
        """
        return self.timeAdvIsoDF
#-----------------------------------------------------------------------------------------------------------
    def get_alpha(self):
        """
        Parameters:   None

        Returns:      list object containing arrays of the conversion values in ascendent order. 
                      The elements are sorted in correspondance with the heating rate list (attribute Beta).
        """
        return self.alpha
#-----------------------------------------------------------------------------------------------------------
    def get_dadt(self):
        """
        Parameters:   None

        Returns:      list object containing arrays of the conversion rates data corresponding 
                      to the conversion values of each element in the attribute alpha. The elements 
                      are sorted in correspondance with the heating rate list (attribute Beta).
        """
        return self.da_dt
#-----------------------------------------------------------------------------------------------------------
    def get_t(self):
        """
        Parameters:   None

        Returns:      list object containing arrays of the time data corresponding to the conversion 
                      values of each element in the attribute alpha. The elements are sorted in 
                      correspondance with the heating rate list (attribute Beta).
        """
        return self.t
#-----------------------------------------------------------------------------------------------------------    
    def get_T(self):
        """
        Parameters:   None

        Returns:      list object containing arrays of the temperature data corresponding to the 
                      conversion values of each element in the attribute alpha. The elements are 
                      sorted in correspondance with the heating rate list (attribute Beta).
        """
        return self.T
#-----------------------------------------------------------------------------------------------------------
    def save_Ea(self, E_Fr= None, E_OFW=None, E_KAS=None, E_Vy=None, E_aVy=None, file_t="xlsx" ):
        """
        Method to save activation energy values calculated with the ActivationEnergy class
         
        Parameters:    E_Fr     : array of activation energies obtained by de Friedman method.

                       E_OFW    : array of activation energies obtained by de OFW method.

                       E_KAS    : array of activation energies obtained by de KAS method.

                       E_Vy     : array of activation energies obtained by de Vyazovkin method.

                       E_aVy    : array of activation energies obtained by de advanced Vyazovkin
                                  method.

                       file_t   :  String. Type of file, can be 'csv' of 'xlsx'.
                                  'xlsx' is the default value.

        returns:       If 'xlsx' is selected, a spreadsheet containg one sheet per experiment
                       containing the values of T, t, and da_dt, plus a sheet containing the 
                       activation energies. 
                       If 'csv' is selected, one 'csv' file per experiment, containing the
                       values of T, t, and da_dt, plus a sheet containing the activation energies. 
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
            raise ValueError("File type not recognized")
#-----------------------------------------------------------------------------------------------------------
    def get_avsT_plot(self):
    """
    Visualization method for alpha vs T

    Parameters:    None

    Returns:       A matplotlib figure plotting conversion vs temperature for
                   each heating rate in attribute Beta.
    """
        for i in range(len(self.DFlis)):
            plt.plot(self.T[i],
                     self.alpha[i],
                     label=str(np.round(self.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[3])
            plt.ylabel(self.DFlis[i].columns[4])
            plt.legend()
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def get_dadtvsT_plot(self):
    """
    Visualization method for da_dt vs T

    Parameters:    None

    Returns:       A matplotlib figure plotting conversion rate vs temperature 
                   for each heating rate in attribute Beta.
    """
        for i in range(len(self.DFlis)):
            plt.plot(self.T[i],
                     self.da_dt[i],
                     label=str(np.round(self.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[3])
            plt.ylabel(self.DFlis[i].columns[5])
            plt.legend()
        return plt.show()

#-----------------------------------------------------------------------------------------------------------
    def get_avst_plot(self):
    """
    Visualization method for alpha vs t

    Parameters:    None

    Returns:       A matplotlib figure plotting conversion vs time for each 
                   heating rate in attribute Beta.
    """
        for i in range(len(self.DFlis)):
            plt.plot(self.t[i],
                     self.alpha[i],
                     label=str(np.round(self.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[0])
            plt.ylabel(self.DFlis[i].columns[4])
            plt.legend()
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def get_dadtvst_plot(self):
    """
    Visualization method for da_dt vs t

    Parameters:    None

    Returns:       A matplotlib figure plotting conversion rate vs time for 
                   each heating rate in attribute Beta.
    """
        for i in range(len(self.DFlis)):
            plt.plot(self.t[i],
                     self.da_dt[i],
                     label=str(np.round(self.Beta[i],decimals=1))+' K/min')
            plt.xlabel(self.DFlis[i].columns[0])
            plt.ylabel(self.DFlis[i].columns[5])
            plt.legend()
        return plt.show()
#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
class ActivationEnergy(object):
    """
	Class that uses the attributes of Dataextraction to compute activation 
    energy values based on five methods: Friedman, FOW, KAS, Vyazovkin and 
    Advanced Vyazovkin. 
    """
    def __init__(self, Beta, TempIsoDF=None, diffIsoDF=None, TempAdvIsoDF=None, timeAdvIsoDF=None):
        """
		Constructor. Defines variables and the constant R=8.314 J/(mol K)

        Parameters:         Beta         : list object containing the values of heating 
                                           rate for each experiment.

                            TempIsoDF    : pandas DataFrame containing the isoconversional
                                           temperatures.  
                      
                            diffIsoDF    : pandas DataFrame containing the isoconversional
                                           derivatives of conversion in respest to time (da_dt).

                            TempAdvIsoDF : pandas DataFrame containing the isoconversional
                                           temperatures, corresponding to evenly spaced values 
                                           of conversion. 
                            
                            timeAdvIsoDF : pandas DataFrame containing the isoconversional
                                           times, corresponding to evenly spaced values of  
                                           conversion.     
        """
        self.E_Fr         = []
        self.E_OFW        = []
        self.E_KAS        = []
        self.E_Vy         = []
        self.E_aVy        = []
        self.Beta         = Beta
        self.logB         = np.log(Beta)
        self.TempIsoDF    = TempIsoDF
        self.diffIsoDF    = diffIsoDF
        self.TempAdvIsoDF = TempAdvIsoDF
        self.timeAdvIsoDF = timeAdvIsoDF
        """ 
        Universal gas constant
        0.0083144626 kJ/(mol*K)
        """
        self.R    = 0.0083144626

#-----------------------------------------------------------------------------------------------------------
    def Fr(self):
        """
        Method to compute the Activation Energy based on the Friedman treatment.

        Parameters:    None

        Returns:       Tuple of arrays:
                       E_Fr   : numpy array containing the activation energy values 
                                obtained by the Friedman method.
             
                       Fr_95e : numpy array containing the 95% trust interval values 
                                obtained by the linear regression in the Friedman method.
                         
                       Fr_b   : numpy array containing the intersection values obtained 
                                by the linear regression in the Friedman method.
        """
        E_Fr      = []
        E_Fr_err  = []
        Fr_b      = []
        diffIsoDF = self.diffIsoDF
        TempIsoDF = self.TempIsoDF
 
        for i in range(0,diffIsoDF.shape[0]):   
            y = np.log(diffIsoDF.iloc[i].values)
            x = 1/(TempIsoDF.iloc[i].values)
            tinv = lambda p, df: abs(t.ppf(p/2, df))
            ts   = tinv(0.05, len(x)-2)
            LR = linregress(x,y)
            E_a_i = -(self.R)*(LR.slope)
            error = -(self.R)*(LR.stderr)
            Fr_b.append(LR.intercept)
            E_Fr_err.append(ts*error)
            E_Fr.append(E_a_i)

        self.E_Fr   = np.array(E_Fr)
        self.Fr_95e = np.array(E_Fr_err)
        self.Fr_b   = np.array(Fr_b) 

        return (self.E_Fr, self.Fr_95e, self.Fr_b)

#-----------------------------------------------------------------------------------------------------------
    def OFW(self):
        """
        Method to compute the Activation Energy based on the Osawa-Flynn-Wall 
	    (OFW) treatment.

        Parameters:    None

        Returns :      Tuple of arrays:
                       E_OFW   : numpy array containing the activation energy values 
                                 obtained by the Ozawa_Flynn-Wall method
             
                       OFW_95e : numpy array containing the 95% trust interval values 
                                 obtained by the linear regression in the 
                                 Ozawa-Flynn-Wall method
        """
        logB       = self.logB
        E_OFW      = []
        E_OFW_err  = []
        TempIsoDF  = self.TempIsoDF
        

        for i in range(TempIsoDF.shape[0]):  
            y = (logB)
            x = 1/(TempIsoDF.iloc[i].values)
            tinv = lambda p, df: abs(t.ppf(p/2, df))
            ts   = tinv(0.05, len(x)-2)
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
        Method to compute the Activation Energy based on the Kissinger-Akahira-Sunose 
        (KAS) treatment.

        Parameters:    None

        Returns :      Tuple of arrays:
                       E_KAS   : numpy array containing the activation energy values 
                                 obtained by the Kissinger-Akahra-Sunose method.
             
                       KAS_95e : numpy array containing the 95% trust interval values 
                                 obtained by the linear regression in the 
                                 Kissinger-Akahra-Sunose method
        """

        logB       = self.logB
        E_KAS      = []
        E_KAS_err  = []
        TempIsoDF  = self.TempIsoDF
       

        for i in range(TempIsoDF.shape[0]):     
            y = (logB)- np.log((TempIsoDF.iloc[i].values)**1.92)
            x = 1/(TempIsoDF.iloc[i].values)
            tinv = lambda p, df: abs(t.ppf(p/2, df))
            ts   = tinv(0.05, len(x)-2)
            LR = linregress(x,y)
            E_a_i = -(self.R)*(LR.slope)
            error = -(self.R)*(LR.stderr)
            E_KAS_err.append(ts*error)
            E_KAS.append(E_a_i)

        self.E_KAS   = np.array(E_KAS)
        self.KAS_95e = np.array(E_KAS_err)
        return self.E_KAS, self.KAS_95e
#-----------------------------------------------------------------------------------------------------------
    def omega(self,E,row,Beta,method = 'senum-yang'):
        """
        Method to calculate the function to minimize for the Vyazovkin method:

        \Omega(Ea) = \sum_{i}^{n}\sum_{j}^{n-1}{[B_{j}{I(E,T_{i})]}/[B_{i}{I(E,T_{j})}]}        

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row    : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        Beta   : list object containing the heatibg rate values 
                                 for each experiment.

                        method : Method to compute the integral temperature.
                                 The available methods are: 'senum-yang' for
                                 the Senum-Yang approximation, 'trapeoid' for
                                 the the trapezoid rule of numerical integration,
                                 and 'quad' for using a technique from the Fortran 
                                 library QUADPACK implemented in the scipy.integrate   
                                 subpackage.

        Returns:        O      : Float. Value of the omega function for the given E.  
        """ 
        #Define integration limits
        IsoDF   = self.TempIsoDF    
        T0      = IsoDF.iloc[0].values   
        T       = IsoDF.iloc[row].values
        #Senum-Yang approximation
        def senum_yang(E):
            x = E/(self.R*T)
            num = x**3 + 18*(x**2) + 88*x + 96
            den = x**4 + 20*(x**3) + 120*(x**2) +240*x +120
            s_y = ((np.exp(-x))/x)*(num/den)
            return (E/self.R)*s_y
        #Trapezoid rule implemented from scipy
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
        #QUAD function implemented from scipy
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
            return np.sum((omega_i))

        elif method == 'trapezoid':
            p = trapezoid(E)
            p_B = p/Beta
            for j in range(len(Beta)):
                y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
                omega_i.append(y)
            return np.sum((omega_i))

        elif method == 'quad':
            p = quad(E)
            p_B = p/Beta
            for j in range(len(Beta)):
                y = p_B[j]*((np.sum(1/(p_B)))-(1/p_B[j]))
                omega_i.append(y)
            return np.sum((omega_i))
        else:
            raise ValueError('method not recognized')

#-----------------------------------------------------------------------------------------------------------
    def visualize_omega(self,row,bounds=(1,300),N=1000,method = 'senum-yang'):
        """
        Method to visualize omega function:

        Parameters:   row    : Int object. Implicit index for the row of conversion in 
                               the pandas DataFrame containing the isoconversional 
                               temperatures.
                             
      
                      bounds : Tuple object containing the lower and upper limit values 
                               for E, to evaluate omega.
 
                      N      : Int. Number of points in the E array for the plot.

                      method : Method to evaluate the temperature integral. The available 
                               methods are: 'senum-yang' for the Senum-Yang approximation,
                               'trapezoid' for the the trapezoid rule of numerical integration,
                               and 'quad' for using a technique from the Fortran library 
                               QUADPACK implemented in the scipy.integrate subpackage.

        Returns:      A matplotlib figure plotting omega vs E. 
        """
        IsoDF   = self.TempIsoDF
        method = method
        E = np.linspace(bounds[0], bounds[1], N)
        O = np.array([float(self.omega(E[i],row,self.Beta,method)) for i in range(len(E))])
        plt.style.use('seaborn')
        plt.plot(E,O,color='teal',label=r'$\alpha$ = '+str(np.round(IsoDF.index[row],decimals=3)))        
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
        plt.legend()

        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def Vy(self, bounds, method='senum-yang'):
        """
        Method to compute the Activation Energy based on the Vyazovkin treatment.

        Parameters:   bounds : Tuple object containing the lower and upper limit values 
                               for E, to evaluate omega.

                      method : Method to evaluate the temperature integral. The available 
                               methods are: 'senum-yang' for the Senum-Yang approximation,
                               'trapezoid' for the the trapezoid rule of numerical integration,
                               and 'quad' for using a technique from the Fortran library 
                               QUADPACK implemented in the scipy.integrate subpackage.

        Returns:      E_Vy   : numpy array containing the activation energy values
                               obtained by the Vyazovkin method. 
        """
        E_Vy       = []
        Beta       = self.Beta 
        IsoDF      = self.TempIsoDF
        
        for k in range(len(IsoDF.index)):
            E_Vy.append(minimize_scalar(self.omega, args=(k,Beta,method),bounds=bounds, method = 'bounded').x)

        self.E_Vy = np.array(E_Vy)
        return self.E_Vy
#-----------------------------------------------------------------------------------------------------------        
    def variance_Vy(self, E, row_i, method = 'senum-yang'):
        """
        Method to calculate the variance of the activation energy E obtained with the Vyazovkin 
        treatment. The variance is computed as:

        S^{2}(E) = {1}/{n(n-1)}\sum_{i}^{n}\sum_{j}^{n-1}{[{J(E,T_{i})]}/[{J(E,T_{j})}]-1}^{2}

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row_i  : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        method : Method to compute the integral temperature.
                                 The available methods are: 'senum-yang' for
                                 the Senum-Yang approximation, 'trapezoid' for
                                 the the trapezoid rule of numerical integration,
                                 and 'quad' for using a technique from the Fortran 
                                 library QUADPACK implemented in the scipy.integrate   
                                 subpackage.

        Returns:        Float object. Value of the variance associated to a given E.  

        --------------------------------------------------------------------------------------------
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """ 

        N   = len(self.Beta)*(len(self.Beta)-1)
        T0  = self.TempIsoDF.iloc[0].values   
        T   = self.TempIsoDF.iloc[row_i].values
            
        def senum_yang(E):
            x = E/(0.008314*T)
            num = x**3 + 18*(x**2) + 88*x + 96
            den = x**4 + 20*(x**3) + 120*(x**2) +240*x +120
            s_y = ((np.exp(-x))/x)*(num/den)
            return (E/0.008314)*s_y
        
        def trapezoid(E):
            x0     = T0
            y0     = np.exp(-E/(0.008314*x0))
            xf     = T
            yf     = np.exp(-E/(0.008314*xf))
            tpz    = [integrate.trapezoid([y0[i],yf[i]],
                                          [x0[i],xf[i]])
                      for i in range(len(T))]
            return np.array(tpz)
        
        def quad(E):
            
            def integral(x,E):
                return np.exp(-E/(0.008314*x))
                
            quad = [integrate.quad(integral,
                                   T0[i],
                                   T[i],
                                   args=(E))[0] 
                    for i in range(len(T))]
            
            return np.array(quad)
                
        if method == 'senum-yang':    
            J = senum_yang(E)
            s = np.array([J[i]/J for i in range(len(J))])
            
            return np.sum((s-1)**2)/N
        
        elif method == 'trapezoid':    
            J = trapezoid(E)
            s = np.array([J[i]/J for i in range(len(J))])
            
            return np.sum((s-1)**2)/N
        
        elif method == 'quad':    
            J = quad(E)
            s = np.array([J[i]/J for i in range(len(J))])
            
            return np.sum((s-1)**2)/N

        else:
            raise ValueError('method not recognized')
#-----------------------------------------------------------------------------------------------------------
    def psi_Vy(self, E, row_i, bounds, method = 'senum-yang'):
        """
        Method to calculate the F distribution to minimize for the Vyazovkin method.
        The distribution is computed as: 

        \Psi(E) = S^{2}(E)/S^{2}_{min}

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row_i  : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        bounds : Tuple object containing the lower and upper limit values 
                                 for E, to evaluate the variance.

                        method : Method to compute the integral temperature.
                                 The available methods are: 'senum-yang' for
                                 the Senum-Yang approximation, 'trapezoid' for
                                 the the trapezoid rule of numerical integration,
                                 and 'quad' for using a technique from the Fortran 
                                 library QUADPACK implemented in the scipy.integrate   
                                 subpackage.

        Returns:        Psi    : Float. Value of the distribution function that sets the lower
                                 and upper confidence limits for E.  
        --------------------------------------------------------------------------------------------
        
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """ 
        
        F      = [161.4, 19, 9.277, 6.388, 5.050, 4.284, 3.787, 3.438, 3.179]
        f      = F[len(self.Beta)-2]
        method = method
        E_min  = minimize_scalar(self.variance_Vy,
                                 bounds=bounds,
                                 args=(row_i,
                                       method), 
                                 method = 'bounded').x
        s_min  = self.variance_Vy(E_min, row_i, method) 
        s      = self.variance_Vy(E, row_i, method)
        
        return (s/s_min) - (f+1)
#-----------------------------------------------------------------------------------------------------------        
    def er_Vy(self, E, row_i, bounds, method = 'senum-yang'):
        """
        Method to compute the error associated to a given activation energy value obtained
        by the vyazovkin method.

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row_i  : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        bounds : Tuple object containing the lower and upper limit values 
                                 for E, to evaluate omega.

                        method : Method to compute the integral temperature.
                                 The available methods are: 'senum-yang' for
                                 the Senum-Yang approximation, 'trapezoid' for
                                 the the trapezoid rule of numerical integration,
                                 and 'quad' for using a technique from the Fortran 
                                 library QUADPACK implemented in the scipy.integrate   
                                 subpackage.

        Returns:        error  : Float. Value of the error associated to a given E.  
        """ 

        method = method

        E_p = np.linspace(5,80,50)
        P = np.array([self.psi_Vy(E_p[i],row_i, bounds, method) for i in range(len(E_p))])
        
        inter_func = interp1d(E_p,
                              P, 
                              kind='cubic',  
                              bounds_error=False, 
                              fill_value="extrapolate")
        
        zeros = np.array([fsolve(inter_func, E-150)[0],
                          fsolve(inter_func, E+150)[0]])
        
        error = np.mean(np.array([abs(E-zeros[0]), abs(E-zeros[1])]))
        
        return error    
#-----------------------------------------------------------------------------------------------------------
    def error_Vy(self, bounds, method = 'senum-yang'):
        """
        Method to calculate the distribution to minimize for the Vyazovkin method.

        Parameters:     bounds   : Tuple object containing the lower and upper limit values 
                                   for E, to evaluate omega.

                        method   : Method to compute the integral temperature. The available 
                                   methods are: 'senum-yang' for the Senum-Yang approximation,
                                   'trapezoid' for the the trapezoid rule of numerical integration,
                                   and 'quad' for using a technique from the Fortran library 
                                   QUADPACK implemented in the scipy.integrate subpackage.

        Returns:        error_Vy : Array of error values associated to the array of activation 
                                   energies obtained by the Vyazovkin method.  
        """         
        bounds = bounds
        method = method
        error_Vy = np.array([self.er_Vy(self.E_Vy[i], i, bounds, method=method) for i in range(len(self.E_Vy))])

        self.error_Vy = error_Vy
        
        return self.error_Vy  
#-----------------------------------------------------------------------------------------------------------              

    def J_Temp(self, E, inf, sup):
        """
        Temperature integral for the Advanced Vyazovkin Treatment.

        Prameters:   E   : Float object. Value for the activation energy to evaluate the integral

                     inf : Inferior integral evaluation limit.

                     sup : Superior integral evaluation limit.

        Returns:     J   : Float. Value of the integral obtained by an analytic expression. Based 
                           on a linear heating rate. 
        """        
        a = E/(self.R)
        b = inf
        c = sup
        J = a*(sp.expi(-a/c)-sp.expi(-a/b)) + c*np.exp(-a/c) - b*np.exp(-a/b)

        return J
#-----------------------------------------------------------------------------------------------------------        
    def J_time(self, E, row_i, col_i, T0, method = 'trapezoid'):
        """
        Time integral for the Advanced Vyazovkin Treatment. Considering a linear heating rate.

        Prameters:   E       : Float object. Value for the activation energy to evaluate the 
                               integral

                     row_i   : Index value for the row of conversion in the pandas DataFrame
                               containing the isoconversional times for evenly spaced conversion 
                               values.
 
                     col_i   : Index value for the column of heating rate in the pandas DataFrame 
                               containing the isoconversional times for evenly spaced conversion 
                               values.

                     T0      : Float. Initial temperature. Must be that corresponding to the 
                               experimental heating rate B.

                     method  : Numerical integration method. Can be 'trapezoid', 'simpson' or 'quad'.
                               The method correspond to those implemented in the scipy.integrate
                               subpackage.

        Returns:     J_t     : Float. Value of the integral obtained by a numerical integration method. 
        """    
        timeAdvIsoDF   = self.timeAdvIsoDF
        B  = self.Beta[col_i]
        t0 = timeAdvIsoDF[timeAdvIsoDF.columns[col_i]][timeAdvIsoDF.index.values[row_i]]
        t  = timeAdvIsoDF[timeAdvIsoDF.columns[col_i]][timeAdvIsoDF.index.values[row_i+1]]
        y0 = np.exp(-E/(self.R*(T0+B*t0)))
        y  = np.exp(-E/(self.R*(T0+B*t)))
            
        if method == 'trapezoid':
            J_t = integrate.trapezoid(y=[y0,y],x=[t0,t])
            return J_t
            
        elif method == 'simpson':
            J_t = integrate.simpson(y=[y0,y],x=[t0,t])
            return J_t
            
        elif method == 'quad':
            def time_int(t,T0,B,E):
                return np.exp(-E/(self.R*(T0+B*t)))
            
            J_t = integrate.quad(time_int,t0,t,args=(T0,B,E))[0]
            return J_t
        else:
            raise ValueError('method not recognized')

#-----------------------------------------------------------------------------------------------------------        
    def adv_omega(self,E, row, var = 'time', method='trapezoid'):
        """
        Function to minimize according to the advanced Vyazovkin treatment:

        \Omega(Ea) = \sum_{i}^{n}\sum_{j}^{n-1}{[{J(E,T(t_{i}))]}/[B_{i}{J(E,T(t_{j}))}]}

        Parameters:   E       : Float object. Value for the activation energy to evaluate 
                                the integral

                      row     : Index value for the row of conversion in the pandas DataFrame
                                containing the isoconversional times for evenly spaced conversion 
                                values.

                      var     : The variable to perform the integral with, it can be either 'time' 
                                or 'Temperature'

                      method  : Numerical integration method. Can be 'trapezoid', 'simpson' or 
                                'quad'. The method correspond to those implemented in the 
                                scipy.integrate subpackage.

        Returns:      O       : Float. Value of the advanced omega function for a given E.
        """
        TempAdvIsoDF = self.TempAdvIsoDF
        timeAdvIsoDF = self.timeAdvIsoDF
        Beta         = self.Beta
        j            = row

        if var == 'Temperature':
            I_x = np.array([self.J_Temp(E,
                                        TempAdvIsoDF[TempAdvIsoDF.columns[i]][TempAdvIsoDF.index[j]],
                                        TempAdvIsoDF[TempAdvIsoDF.columns[i]][TempAdvIsoDF.index[j+1]]) 
                            for i in range(len(TempAdvIsoDF.columns))])
            I_B = I_x/Beta
            omega_i = np.array([I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k])) for k in range(len(Beta))])
            O = np.array(np.sum((omega_i)))
            return O
  
        elif var == 'time':
            I_B = np.array([self.J_time(E,
                                        row,
                                        i,
                                        TempAdvIsoDF.iloc[0][i],
                                        method) 
                            for i in range(len(timeAdvIsoDF.columns))])

            omega_i = np.array([I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k])) for k in range(len(Beta))])
            O = np.array(np.sum((omega_i)))
            return O        
#-----------------------------------------------------------------------------------------------------------
    def visualize_advomega(self,row,var='time',bounds=(1,300),N=1000, method='trapezoid'):
        """
        Method to visualize adv_omega function. 

        Parameters:   row     : Index value for the row of conversion in the pandas DataFrame
                                containing the isoconversional times or temperatures.
                     
                      var     : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'. Default 'time'.
      
                      bounds  : Tuple object containing the lower limit and the upper limit values 
                                of E, for evaluating adv_omega. Default (1,300).

                      N       : Int. Number of points in the E array for the plot. Default 1000.

                      method  : Numerical integration method. Can be 'trapezoid', 'simpson'
                                or 'quad'. The method correspond to those implemented in 
                                the scipy.integrate subpackage. Default 'trapezoid'.


        Returns:      A matplotlib plot of adv_omega vs E 
        """
        TempAdvIsoDF = self.TempAdvIsoDF
        timeAdvIsoDF = self.timeAdvIsoDF
        Beta         = self.Beta

        E = np.linspace(bounds[0], bounds[1], N)
        O = np.array([float(self.adv_omega(E[i],row,var,method)) for i in range(len(E))])
        plt.style.use('seaborn')
        plt.plot(E,O,color='teal',label=r'$\alpha$ = '+str(np.round(timeAdvIsoDF.index[row],decimals=3)))
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
        plt.legend()
    
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
    def aVy(self,bounds, var='time', method='trapezoid'):
        """
        Method to compute the Activation Energy based on the Advanced Vyazovkin treatment.
        
        Parameters:   bounds : Tuple object containing the lower limit and the upper 
                               limit values of E, for evaluating omega.

                      T      : List object containing the experimental temperatures. 
                               Must be those corresponding to the experimental heating 
                               rate.

                      var    : The variable to perform the integral with, it can be either
                               'time' or 'Temperature'

                      method  : Numerical integration method. Can be 'trapezoid', 'simpson'
                                or 'quad'. The method correspond to those implemented in 
                                the scipy.integrate subpackage. Default 'trapezoid'.

        Returns:      E_Vy   : numpy array containing the activation energy values
                               obtained by the Vyazovkin method. 
        """
        TempAdvIsoDF = self.TempAdvIsoDF
        timeAdvIsoDF = self.timeAdvIsoDF
        Beta         = self.Beta

        E_aVy        = [minimize_scalar(self.adv_omega,bounds=bounds,args=(k,var,method), method = 'bounded').x 
                        for k in range(len(timeAdvIsoDF.index)-1)]

        self.E_aVy   = np.array(E_aVy)

        return self.E_aVy
#-----------------------------------------------------------------------------------------------------------        
    def variance_aVy(self, E, row_i, var = 'time', method = 'trapezoid'):
        """
        Method to calculate the variance of the activation energy E obtained with the Vyazovkin 
        treatment. The variance is computed as:

        S^{2}(E) = {1}/{n(n-1)}\sum_{i}^{n}\sum_{j}^{n-1}{[{J(E,T(t_{i}))]}/[{J(E,T(t_{j}))}]-1}^{2}

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row_i  : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        var    : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'

                        method : Numerical integration method. Can be 'trapezoid', 'simpson'
                                 or 'quad'. The method correspond to those implemented in 
                                 the scipy.integrate subpackage. Default 'trapezoid'.

        Returns:        Float object. Value of the variance associated to a given E.  

        --------------------------------------------------------------------------------------------
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """

        N = len(self.Beta)*(len(self.Beta)-1)
        
        if var == 'time':
             
            inf = self.timeAdvIsoDF.index.values[row_i] 
            sup = self.timeAdvIsoDF.index.values[row_i+1]
            T0  = self.TempIsoDF.iloc[0]
                    
            J = np.array([self.J_time(E, row_i, i, T0[i], method) for i in range(len(self.Beta))])     
            s = np.array([J[i]/J for i in range(len(J))])
            
            return np.sum((s-1)**2)/N
            
        elif var == 'Temperature':
            
            inf = self.TempAdvIsoDF.index.values[row_i] 
            sup = self.TempAdvIsoDF.index.values[row_i+1]
        
            
            J = [self.J_Temp(E, 
                            self.TempAdvIsoDF[self.TempAdvIsoDF.columns[i]][inf], 
                            self.TempAdvIsoDF[self.TempAdvIsoDF.columns[i]][sup]) 
                 for i in range(len(self.Beta))]
            s = np.array([J[i]/J for i in range(len(J))])
            
            return np.sum((s-1)**2)/N

        else:
            raise ValueError('variable not valid')

#-----------------------------------------------------------------------------------------------------------        
    def psi_aVy(self, E, row_i, bounds, var = 'time', method = 'trapezoid'):
        """
        Method to calculate the F distribution to minimize for the Vyazovkin method.
        The distribution is computed as: 

        \Psi(E) = S^{2}(E)/S^{2}_{min}

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row_i  : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        bounds : Tuple object containing the lower and upper limit values 
                                 for E, to evaluate the variance.

                        var    : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'

                        method : Numerical integration method. Can be 'trapezoid', 'simpson'
                                 or 'quad'. The method correspond to those implemented in 
                                 the scipy.integrate subpackage. Default 'trapezoid'.

        Returns:        Psi    : Float. Value of the distribution function that sets the lower
                                 and upper confidence limits for E.  
        --------------------------------------------------------------------------------------------
        
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """         
        F      = [161.4, 19, 9.277, 6.388, 5.050, 4.284, 3.787, 3.438, 3.179]
        f      = F[len(self.Beta)-2]
        var    = var
        method = method
        E_min  = minimize_scalar(self.variance_aVy,
                                 bounds=bounds,
                                 args=(row_i,
                                       var,
                                       method), 
                                 method = 'bounded').x
        s_min  = self.variance_aVy(E_min, row_i, var, method) 
        s      = self.variance_aVy(E, row_i, var, method)
        
        return (s/s_min) - (f+1)
    
#-----------------------------------------------------------------------------------------------------------        
    def er_aVy(self, E, row_i, bounds, var = 'time', method = 'trapezoid'):
        """
        Method to compute the error associated to a given activation energy value obtained
        by the vyazovkin method.

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row_i  : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        bounds : Tuple object containing the lower and upper limit values 
                                 for E, to evaluate adv_omega.

                        var    : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'

                        method : Numerical integration method. Can be 'trapezoid', 'simpson'
                                 or 'quad'. The method correspond to those implemented in 
                                 the scipy.integrate subpackage. Default 'trapezoid'.

        Returns:        error  : Float. Value of the error associated to a given E.  
        """        
        var    = var
        method = method

        E_p = np.linspace(5,80,50)
        P = np.array([self.psi_aVy(E_p[i],row_i, bounds, var, method) for i in range(len(E_p))])
        
        inter_func = interp1d(E_p,
                          P, 
                          kind='cubic',  
                          bounds_error=False, 
                          fill_value="extrapolate")
        
        zeros = np.array([fsolve(inter_func, E-150)[0],
                          fsolve(inter_func, E+150)[0]])
        
        error = np.mean(np.array([abs(E-zeros[0]), abs(E-zeros[1])]))
        
        return error
#-----------------------------------------------------------------------------------------------------------        
    def error_aVy(self, bounds, var = 'time', method = 'trapezoid'):
        """
        Method to calculate the distribution to minimize for the Vyazovkin method.

        Parameters:     bounds   : Tuple object containing the lower and upper limit values 
                                   for E, to evaluate adv_omega.

                        var    : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'

                        method : Numerical integration method. Can be 'trapezoid', 'simpson'
                                 or 'quad'. The method correspond to those implemented in 
                                 the scipy.integrate subpackage. Default 'trapezoid'.

        Returns:        error_aVy : Array of error values associated to the array of activation 
                                    energies obtained by the Vyazovkin method.  
        """ 
        bounds = bounds
        var    = var
        method = method
        error_aVy = np.array([self.er_aVy(self.E_aVy[i], i, bounds, var=var, method=method) for i in range(len(self.E_aVy))])

        self.error_aVy = error_aVy
        
        return self.error_aVy
#-----------------------------------------------------------------------------------------------------------        
    def prediction(self, E = None, B = 1, T0 = 298.15, Tf=1298.15):
        """
        Method to calculate a kinetic prediction, based on an isoconversional 
        activation energy

        Parameters:   E  : numpy array of the activation energy values to use for
                           the prediction.

                      B  : Float. Value of the heating rate for the prediction.

                      T0 : Float. Initial temperature, in Kelvin, for the prediction.

                      Tf : Float. Final temperature, in Kelvin, for the prediction.

        Returns:      a  : numpy array containing the predicted conversion values.

                      T  : numpy array cppntaining the temperature values corresponding 
                           to the predicted conversion.

                      t  : numpy array cppntaining the time values corresponding to the 
                           predicted conversion.
        """
        b      = np.exp(self.Fr_b)
        a_pred = [0]
        T      = np.linspace(T0,Tf,len(b))
        t      =  (T-T0)/B
        dt     =  t[1]-t[0]
        for i in range(len(b)-1):
            a = a_pred[i] + b[i]*np.exp(-(E[i]/(self.R*(T0+B*t[i]))))*dt
            a_pred.append(a)

        a_pred      = np.array(a_pred)
        self.a_pred = a_pred
       
        return (self.a_pred, T, t)


