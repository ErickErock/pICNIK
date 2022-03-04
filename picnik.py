#!/usr/bin/python3.7
"""
This module has two classes: DataExtraction and ActivationEnergy. 
DataExtraction reads csv files and creates pandas.DataFrames according 
to the isoconversional principle. ActivationEnergy computes the activation
energy with five implemented isoconversional methods: Friedman (Fr), 
Ozawa-Flynn-Wall(OFW), Kissinger-Akahira-Sunos (KAS) and the method 
developed by Vyazovkin (Vy, aVy).
"""
#Dependencies
import numpy as np
import pandas as pd
from   scipy.interpolate import interp1d
from   scipy.optimize    import minimize_scalar
import scipy.special     as     sp
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import integrate
import derivative
from scipy.optimize import fsolve


#-----------------------------------------------------------------------------------------------------------
class DataExtraction(object):
    """
    Extractor to manipulate raw data to create lists and Data Frames 
    that will be used to compute the Activation Energy.
    """
    def __init__(self):
        """
        Constructor. 

        Parameters:    None

        Notes:         It only defines variables.
        """
        self.DFlis          = []              #list of DataFrames containing data
        self.seg_DFlis      = []              #list of DataFrames segmented by temperature 
        self.Beta           = []              #list of heating rates
        self.BetaCC         = []              #list of correlation coefficient for T vs t
        self.files          = []              #list of files containing raw data
        self.da_dt          = []              #list of experimental conversion rates 
        self.T              = []              #list of experimental temperature in Kelvin
        self.T0             = []              #list of experimental inicial temperature in Kelvin
        self.t              = []              #list off experimental time
        self.alpha          = []              #list of experimental conversion
        self.TempIsoDF      = pd.DataFrame()  #Isoconversional temperature DataFrame
        self.timeIsoDF      = pd.DataFrame()  #Isoconversional time DataFrame
        self.diffIsoDF      = pd.DataFrame()  #Isoconversional conversion rate DataFrame
        self.TempAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional temperature DataFrame
        self.timeAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional time DataFrame
#-----------------------------------------------------------------------------------------------------------    
    def read_files(self, flist, encoding='utf8'):
        """ 
        Reads each TGA file as a pandas DataFrame and calculates de heating rate 
        
        Parameters:    flist : list object containing the paths of the files to be used.
        
                       encoding : The available encodings for pandas.read_csv() method. Includes but not limited 
                                  to 'utf8', 'utf16','latin1'. For more information on the python standar encoding:
                                  (https://docs.python.org/3/library/codecs.html#standard-encodings)
        """
    
        print("Files to be used: \n{}\n ".format(flist))
        DFlis         =   self.DFlis
        Beta          =   self.Beta
        BetaCorrCoeff =   self.BetaCC
        T0            =   self.T0            
        print(f'Reading files and creating DataFrames...\n')
        for item in flist:
            #csv files can use a tab or a coma as separator.
            try:
                DF = pd.read_csv(item,  sep = '\t', encoding = encoding)
                #stores the initial temperature of the ith experiment
                T0.append(DF[DF.columns[1]][0]+273.15)                       
                #computes the mass loss percentage
                DF['%m'] = 100*(DF[DF.columns[2]]/DF[DF.columns[2]][0])
                #creates a column for the temperature in Kelvin
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15      
                #computes the heating rate with a Savitzki-Golay filter
                dTdt = derivative.dxdt(DF['Temperature [K]'].values,             
                                       DF[DF.columns[0]].values,
                                       kind="savitzky_golay", 
                                       order=3,
                                       left=0.5,
                                       right=0.5)
                DF['dT/dt$'] = DF[DF.columns[0]]
                DF['dT/dt$'] = dTdt         
                #computes the differential thermogram with a Savitzki-Golay filter                                    
                dwdt = derivative.dxdt(DF[DF.columns[2]].values,             
                                       DF[DF.columns[0]].values,          
                                       kind="savitzky_golay",
                                       order=3,
                                       left=0.5,
                                       right=0.5)
                DF['dw/dt'] = DF[DF.columns[0]]
                DF['dw/dt'] = dwdt 
                
                #computes the heating rate
                LR = linregress(DF[DF.columns[0]],                            
                                DF[DF.columns[1]])

                BetaCorrCoeff.append(LR.rvalue)
                Beta.append(LR.slope)
                DFlis.append(DF)

            except IndexError:
                DF = pd.read_csv(item,  sep = ',', encoding = encoding)
                T0.append(DF[DF.columns[1]][0]+273.15)
                DF['%m'] = 100*(DF[DF.columns[2]]/DF[DF.columns[2]][0])
                #creates a column for the temperature in Kelvin
                DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15      
                #computes the differential thermogram with a Savitzki-Golay filter
                dTdt = derivative.dxdt(DF[DF.columns[1]].values,
                                       DF[DF.columns[0]].values,
                                       kind="savitzky_golay", 
                                       order=3,
                                       left=0.5,
                                       right=0.5)
                DF['dT/dt$'] = DF[DF.columns[0]]
                DF['dT/dt$'] = dTdt
                dwdt = derivative.dxdt(DF[DF.columns[2]].values,
                                       DF[DF.columns[0]].values,
                                       kind="savitzky_golay",
                                       order=3,
                                       left=0.5,
                                       right=0.5)
                DF['dw/dt'] = DF[DF.columns[0]]
                DF['dw/dt'] = dwdt 
                
                #computes the heating rate
                LR = linregress(DF[DF.columns[0]],
                                DF[DF.columns[1]])

                BetaCorrCoeff.append(LR.rvalue)
                Beta.append(LR.slope)
                DFlis.append(DF)

        self.DFlis  = DFlis                     #List of the DataFrames constructed
        self.Beta   = np.array(Beta)            #Array of heating rates in ascendent order
        self.BetaCC = np.array(BetaCorrCoeff)   #Array of correlation coefficients for the heating rates
        self.T0     = np.array(T0)              #Array of experimental initial temperatures

        print(f'The computed heating rates are:\n')
        for i in range(len(Beta)):
            print(f'{Beta[i]:.2f} K/min')
        return self.Beta, self.T0 
#-----------------------------------------------------------------------------------------------------------
    def Conversion(self,T0,Tf):
        """
        Calculates the conversion values for a given temperature range. 
        Not all experimental points are suitable for the isoconversional 
        analysis, so a temperature analysis range must be selected based 
        on the thermal profile of the sample.
        
        Parameters:    T0: Initial temperature in Kelvin of the interval where the process to study is.
                         
                       Tf: Final temperature in Kelvin of the interval where the process to study is.

        Returns:       A plot of the temperature range to be used in the analysis.
                       
        """
        DFlist = self.DFlis
        NDFl = self.seg_DFlis
        print('The temperature range was set to ({0:0.1f},{1:0.1f}) K'.format((T0),(Tf)))
        print(f'Computing conversion values...')
        for item in DFlist:
                #filters the DataFrames based on the temperature limits 
                item = item.loc[(item['Temperature [K]'] > T0) & (item['Temperature [K]'] < Tf)]     
                item = item.reset_index(drop=True)
                #calculates the conversion                         
                item['alpha'] = (item[item.columns[2]][0]-item[item.columns[2]])/(item[item.columns[2]][0]-item[item.columns[2]][item.shape[0]-1])
                #computes the cnversion rate with a Savitzki-Golay filter
                dadt = derivative.dxdt(item['alpha'].values,
                                       item[item.columns[0]].values,
                                       kind="savitzky_golay", 
                                       order=3,
                                       left=0.5,
                                       right=0.5)
                item['da/dt'] = item[item.columns[0]]
                item['da/dt'] = dadt
                NDFl.append(item)
        alpha = self.alpha
        T = self.T
        t = self.t
        da_dt = self.da_dt

        #To create the Isoconversional DataFrames interpolation is needed. 
        #In order to make the interpolation the x values must be strictly in ascending order.
        #The next block of code evaluates if the i-th value is bigger than the i-1-th, if so, 
        #the value is appended to the corresponding list. 
        for i in range(len(NDFl)):
            #The initial values are those of the lower limit of the temperature range.
            a = [NDFl[i]['alpha'].values[0]]
            Temp = [NDFl[i]['Temperature [K]'].values[0]]
            time = [NDFl[i][DFlist[i].columns[0]].values[0]]
            diff = [NDFl[i]['da/dt'].values[1]] 
            for j in range(len(NDFl[i]['alpha'].values)):
                if NDFl[i]['alpha'].values[j] == a[-1]:
                    pass
                #If the i-th value is bigger than the i-1-th
                #its corresponding values of time, temperature 
                #and conversion rate and itself are stored
                #in a corresponding list.
                elif NDFl[i]['alpha'].values[j] > a[-1]:
                    a.append(NDFl[i]['alpha'].values[j])
                    Temp.append(NDFl[i]['Temperature [K]'].values[j])
                    time.append(NDFl[i][NDFl[i].columns[0]].values[j])
                    diff.append(NDFl[i]['da/dt'].values[j])
                else:
                    pass
            alpha.append(np.array(a))
            T.append(np.array(Temp))
            t.append(np.array(time))
            da_dt.append(np.array(diff))
        print(f'Done')
        
        self.seg_DFlis = NDFl      #list of segmented DataFrames
        self.alpha     = alpha     #list of arrays of conversion values for each heating rate
        self.T         = T         #list of arrays of temperatures corresponding to a conversion value
        self.t         = t         #list of arrays of temperatures corresponding to a conversion value
        self.da_dt     = da_dt

        plt.style.use('tableau-colorblind10')

        markers = ["o","v","x","1","s","^","p","<","2",">"]
        #Plot of the thermograms showing the anaysis range.
        for i in range(len(DFlist)):
            plt.plot(DFlist[i]['Temperature [K]'].values[::40],           #Temperature in Kelvin
                     DFlist[i]['%m'].values[::40],                        #mass loss percentage
                     marker = markers[i],
                     linestyle = '--',
                     label=r'$\beta=$'+str(np.round(self.Beta[i],decimals=2))+' K/min',
                     alpha=0.75)
        plt.axvline(x=(T0),alpha=0.8,color='red',ls='--',lw=1.2)         #temperature lower limit
        plt.axvline(x=(Tf),alpha=0.8,color='red',ls='--',lw=1.2)         #temperature upper limit
        plt.ylabel('m%')
        plt.xlabel('T [K]')
        plt.xlim((T0-20),(Tf+20)) 
        plt.legend(frameon=True)
        plt.grid(True)

        plt.show()
#-----------------------------------------------------------------------------------------------------------
    def Isoconversion(self, advanced = False, method='points', N = 1000, d_a = 0.001):    
        """
        Constructs the isoconversional DataFrames.

        Parameters:             advanced: Boolean value. If set to True the advanced isoconverional 
                                          DataFrames will be constructed.
     
                                method:   String. 'points' or 'step'. In case of setting advanced to 
                                          True the conversion array can be constructed con the linspace 
                                          or arange functions of numpy. 'points' will call for linspace
                                          while 'step' will call for arange.

                                N:        The number of points in the conversion array If method is set 
                                          to 'points'.

                                d_a:      The size of the step from the i-th to the i+1-th value in the
                                          conversion array If method is set to 'step'.

        Returns:                pandas.DataFrame objects: Temperatures Dataframe, times DataFrame, conversion
                                rates DataFrame. If advanced is to True it also returns a Temperatures and times
                                DataFrames for the aadvanced method of Vyazovkin (aVy method in ActivationEnergy).                     
               
        """
        
        alpha = self.alpha
        T     = self.T
        t     = self.t
        da_dt = self.da_dt
        Beta  = self.Beta
        
        TempIsoDF    = self.TempIsoDF      
        timeIsoDF    = self.timeIsoDF     
        diffIsoDF    = self.diffIsoDF      
        TempAdvIsoDF = self.TempAdvIsoDF   
        timeAdvIsoDF = self.timeAdvIsoDF  
        #The experimental set with the least points is selected as conversion 
        #array for the isoconversional coomputations because all the other data sets
        #have more points to interpolate a reliable function for the conversion array
        alps = np.array(alpha[-1])
        print(f'Creating Isoconversion DataFrames...')
        #The time, temperature and conversion rate values corresponding to conversion array
        #selected are pass atrightforward to the corresponding isoconversional DataFrame
        TempIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(T[-1], decimals = 4)
        timeIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(t[-1], decimals = 4)        
        diffIsoDF['HR '+str(np.round(Beta[-1], decimals = 1)) + ' K/min'] = np.round(da_dt[-1], decimals = 4)        

        for i in range(len(Beta)-1):
            #The interpolation functions to compute isoconversional values are constructed 
            #as cubic splines with the scipy.interpolate.interp1d function
            inter_func = interp1d(alpha[i],
                                  t[i], 
                                  kind='cubic', 
                                  bounds_error=False, 
                                  fill_value="extrapolate")
            #A column is added to the isoconversional DataFrames for each heating rate 
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

        #Sorting the columns in ascending order
        colnames          = TempIsoDF.columns.tolist()
        colnames          = colnames[1:] + colnames[:1]
        #Asigning the values of the conversion array as index for the DataFrames
        TempIsoDF.index   = alpha[-1]
        TempIsoDF         = TempIsoDF[colnames]       #Isoconversional DataFrame of temperature
        timeIsoDF.index   = alpha[-1]
        timeIsoDF         = timeIsoDF[colnames]       #Isoconversional DataFrame of time
        diffIsoDF.index   = alpha[-1]
        diffIsoDF         = diffIsoDF[colnames]       #Isoconversional DataFrame of conversion rate
        
        self.TempIsoDF  = TempIsoDF 
        self.timeIsoDF  = timeIsoDF
        self.diffIsoDF  = diffIsoDF
        
        if advanced == True:
        #Conversion array based on the number of points.
            if method == 'points':
                adv_alps, d_a = np.linspace(alpha[-1][0],alpha[-1][-1],N,retstep=True)
        #Conversion array based on the \Delta\alpha value
            elif method == 'step':
                adv_alps = np.arange(alpha[-1][0],alpha[-1][-1],d_a)
            else:
                raise ValueError('Method not recognized')

            for i in range(0,len(Beta)):
                #New interpolation functions with the advanced conversion array
                inter_func = interp1d(alpha[i], 
                                      T[i],
                                      kind='cubic', 
                                      bounds_error=False, 
                                      fill_value="extrapolate")
                TempAdvIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func(adv_alps), decimals = 4)

                inter_func2 = interp1d(alpha[i], 
                                       t[i],
                                       kind='cubic', bounds_error=False, 
                                       fill_value="extrapolate")
                timeAdvIsoDF['HR '+str(np.round(Beta[i], decimals = 1)) + ' K/min'] = np.round(inter_func2(adv_alps), decimals = 4)
            
            timeAdvIsoDF.index = adv_alps
            TempAdvIsoDF.index = adv_alps

            self.TempAdvIsoDF = TempAdvIsoDF      #Isoconversional DataFrame of temperature for the advanced Vyazovkin method (aVy)
            self.timeAdvIsoDF = timeAdvIsoDF      #Isoconversional DataFrame of time for the advanced Vyazovkin method (aVy)
            self.d_a          = d_a               #Size of the \Delta\alpha step
        else:
            pass
        
        print(f'Done')

        return self.TempIsoDF, self.timeIsoDF, self.diffIsoDF, self.TempAdvIsoDF, self.timeAdvIsoDF
#-----------------------------------------------------------------------------------------------------------        
    def get_beta(self):
        """
        Getter for the heating rates.

        Parameters:   None

        Returns:      array object containing the experimental heating rate sorted 
                      in ascendent order obtained from a linear regression of T vs t.
        """
        return self.Beta
#-----------------------------------------------------------------------------------------------------------
    def get_betaCC(self):
        """
        Getter for the correlation coefficient of the heating rates.

        Parameters:   None

        Returns:      list object containing the experimental T vs t correlation coefficient
                      obtained from a linear regression, sorted in correspondance with the 
                      heating rate list (attribute Beta).
        """
        return self.BetaCC
#-----------------------------------------------------------------------------------------------------------       
    def get_DFlis(self):
        """
        Getter of the list containing the DataFrames of the experimental runs.

        Parameters:   None

        Returns:      list object containing the DataFrames with the experimental data, sorted 
                      in correspondance with the heating rate list (attribute Beta).
        """
        return self.DFlis
#-----------------------------------------------------------------------------------------------------------
    def get_TempIsoDF(self):
        """
        Getter for the Temperatures DataFrame.
  
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
        Getter for the times DataFrame.

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
        Getter for the conversion rates DataFrame.

        Parameters:   None

        Returns:      DataFrame of isoconversional conversion rates. The index is the set of conversion 
                      values from the experiment with the less data points (which correspond to the smallest 
                      heating rate). The columns are isoconversional conversion rates, sorted in heating 
                      rate ascendent order from left to right.
        """
        return self.timeIsoDF
#-----------------------------------------------------------------------------------------------------------
    def get_TempAdvIsoDF(self):
        """
        Getter for the Temperatures DataFrame for the advenced method of Vyazovkin (aVy).
 
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
        Getter for the times DataFrame for the advenced method of Vyazovkin (aVy).

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
        Getter for the list of arrays containig conversion values.

        Parameters:   None

        Returns:      list object containing arrays of the conversion values in ascendent order. 
                      The elements are sorted in correspondance with the heating rate list (attribute Beta).
        """
        return self.alpha
#-----------------------------------------------------------------------------------------------------------
    def get_dadt(self):
        """
        Getter for the list of arrays containig conversion rate values corresponding to the alpha arrays.

        Parameters:   None

        Returns:      list object containing arrays of the conversion rates data corresponding 
                      to the conversion values of each element in the attribute alpha. The elements 
                      are sorted in correspondance with the heating rate list (attribute Beta).
        """
        return self.da_dt
#-----------------------------------------------------------------------------------------------------------
    def get_t(self):
        """
        Getter for the list of arrays containig time values corresponding to the alpha arrays.

        Parameters:   None

        Returns:      list object containing arrays of the time data corresponding to the conversion 
                      values of each element in the attribute alpha. The elements are sorted in 
                      correspondance with the heating rate list (attribute Beta).
        """
        return self.t
#-----------------------------------------------------------------------------------------------------------    
    def get_T(self):
        """
        Getter for the list of arrays containig temperature values corresponding to the alpha arrays.

        Parameters:   None

        Returns:      list object containing arrays of the temperature data corresponding to the 
                      conversion values of each element in the attribute alpha. The elements are 
                      sorted in correspondance with the heating rate list (attribute Beta).
        """
        return self.T
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
            plt.xlabel('T [K]')
            plt.ylabel(r'$\alpha$')
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
            plt.xlabel('T [K]')
            plt.ylabel(r'$\text{d}\alpha/\text{d}t [min$^{-1}$]')
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
            plt.ylabel('$\alpha$')
            plt.legend()
        return plt.show()
#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
class ActivationEnergy(object):
    """
	Uses the attributes of Dataextraction to compute activation energy values based on five methods: 
    Friedman, FOW, KAS, Vyazovkin and Advanced Vyazovkin. 
    """
    def __init__(self, Beta, T0, TempIsoDF=None, diffIsoDF=None, TempAdvIsoDF=None, timeAdvIsoDF=None):
        """
		Constructor. Defines variables and the constant R=8.314 J/(mol K)

        Parameters:         Beta         : array object containing the values of heating 
                                           rate for each experiment.
 
                            T0           : array of initial experimental temperatures.

                            TempIsoDF    : pandas DataFrame containing the isoconversional
                                           temperatures.  
                      
                            diffIsoDF    : pandas DataFrame containing the isoconversional
                                           conversion rate (da_dt).

                            TempAdvIsoDF : pandas DataFrame containing the isoconversional
                                           temperatures, corresponding to evenly spaced values 
                                           of conversion. 
                            
                            timeAdvIsoDF : pandas DataFrame containing the isoconversional
                                           times, corresponding to evenly spaced values of  
                                           conversion.     
        """
        
        self.Beta         = Beta             #Array of heating rates
        self.logB         = np.log(Beta)     #Array of log10(heating rate) 
        self.TempIsoDF    = TempIsoDF        #Isoconversional DataFrame of temperatures
        self.diffIsoDF    = diffIsoDF        #Isoconversional DataFrames of conversion rates
        self.TempAdvIsoDF = TempAdvIsoDF     #Isoconversional DataFrame of temperatures for the advanced Vyazovkin method (aVy)
        self.timeAdvIsoDF = timeAdvIsoDF     #Isoconversional DataFrame of times for the advanced Vyazovkin method (aVy)
        self.T0           = T0               #Array of initial experimental temperatures
        self.E_Fr         = []               #Container for the Friedmann (Fr) method results
        self.E_OFW        = []               #Container for the OFW method (OFW) results
        self.E_KAS        = []               #Container for the KAS method (KAS) results
        self.E_Vy         = []               #Container for the Vyazovkin method (Vy) results
        self.E_aVy        = []               #Container for the advanced Vyazovkin method (aVy)results

        self.R            = 0.0083144626     #Universal gas constant 0.0083144626 kJ/(mol*K)

#-----------------------------------------------------------------------------------------------------------
    def Fr(self):
        """
        Computes the Activation Energy based on the Friedman treatment.
        \ln{(d\alpha/dt)}_{\alpha ,i} = \ln{[A_{\alpha}f(\alpha)]}-\frac{E_{\alpha}}{RT_{\alpha ,i}}

        Parameters:    None

        Returns:       Tuple of arrays:
                       E_Fr   : numpy array containing the activation energy values 
                                obtained by the Friedman method.
             
                       Fr_95e : numpy array containing the standard deviation of the.
                                activation energies obtained by the Friedman method.
                       Fr_b   : numpy array containing the intersection values obtained 
                                by the linear regression in the Friedman method.
        ----------------------------------------------------------------------------------
        Reference:     H. L. Friedman, Kinetics of thermal degradation of char-forming plastics
                       from thermogravimetry. application to a phenolic plastic, in: Journal of
                       polymer science part C: polymer symposia, Vol. 6, Wiley Online Library,
                       1964, pp. 183–195.
        """
        E_Fr      = []
        E_Fr_err  = []
        Fr_b      = []
        diffIsoDF = self.diffIsoDF
        TempIsoDF = self.TempIsoDF
        print(f'Friedman method: Computing activation energies...')
        for i in range(0,diffIsoDF.shape[0]):
        #Linear regression over all the conversion values in the isoconversional Dataframes
            y     = np.log(diffIsoDF.iloc[i].values)             #log(da_dt)
            x     = 1/(TempIsoDF.iloc[i].values)                 #1/T
            LR    = linregress(x,y)
            E_a_i = -(self.R)*(LR.slope)                         #Activation Energy

            E_Fr.append(E_a_i)            
            Fr_b.append(LR.intercept)                            #ln[Af(a)]
            error = -(self.R)*(LR.stderr)                        #Standard deviation of the activation energy

            E_Fr_err.append(error)

        E_Fr   = np.array(E_Fr)
        Fr_e = np.array(E_Fr_err)
        Fr_b   = np.array(Fr_b)
        #Tuple with the results: Activation energy, Standard deviation and ln[Af(a)]
        self.E_Fr =  (E_Fr, Fr_e, Fr_b)                          
        print(f'Done.')
        return self.E_Fr

#-----------------------------------------------------------------------------------------------------------
    def OFW(self):
        """
        Computes the Activation Energy based on the Osawa-Flynn-Wall (OFW) treatment.
        \ln{\beta_{i}} = cnt - 1.052\frac{E_{\alpha}}{RT_{\alpha ,i}}

        Parameters:    None

        Returns :      Tuple of arrays:
                       E_OFW   : numpy array containing the activation energy values 
                                 obtained by the Ozawa_Flynn-Wall method
             
                       OFW_s   : numpy array containing the standard deviation of the 
                                 activation energy values obtained by the linear regression 
                                 in the Ozawa-Flynn-Wall method
        -----------------------------------------------------------------------------------------------
        References:   T. Ozawa, A new method of analyzing thermogravimetric data, Bulletin
                      of the chemical society of Japan 38 (11) (1965) 1881–1886.

                      J. H. Flynn, L. A. Wall, A quick, direct method for the determination
                      of activation energy from thermogravimetric data, Journal of Polymer
                      Science Part B: Polymer Letters 4 (5) (1966) 323–328.
        """
        logB       = self.logB
        E_OFW      = []
        E_OFW_err  = []
        TempIsoDF  = self.TempIsoDF
        print(f'Ozawa-Flynn-Wall method: Computing activation energies...')        
        for i in range(TempIsoDF.shape[0]):  
        #Linear regression over all the conversion values in the isoconversional Dataframes
            y = (logB)                                           #log(\beta)
            x = 1/(TempIsoDF.iloc[i].values)                     #1/T
            LR = linregress(x,y)
            E_a_i = -(self.R/1.052)*(LR.slope)                   #Activation energy
            error = -(self.R/1.052)*(LR.stderr)                  #Standard deviation of the activation energy
            E_OFW_err.append(error)
            E_OFW.append(E_a_i)

        E_OFW   = np.array(E_OFW)
        OFW_s = np.array(E_OFW_err)   
        #Tuple with the results: Activation energy, Standard deviation
        self.E_OFW   = (E_OFW, OFW_s)
        print(f'Done.')
        return self.E_OFW
#-----------------------------------------------------------------------------------------------------------
    def KAS(self):
        """
        Computes the Activation Energy based on the Kissinger-Akahira-Sunose (KAS) treatment.
        \ln{\frac{\beta_{i}}{T^{2}_{\alpha ,i}} = cnt - \frac{E_{\alpha}}{RT_{\alpha ,i}}
         
        Parameters:    None

        Returns :      Tuple of arrays:
                       E_KAS   : numpy array containing the activation energy values 
                                 obtained by the Kissinger-Akahra-Sunose method.
             
                       KAS_s   : numpy array containing the standard deviation of the 
                                 activation energy values obtained by the linear regression 
                                 in the Kissinger-Akahra-Sunose method.
        ---------------------------------------------------------------------------------------
        Reference:     H. E. Kissinger, Reaction kinetics in differential thermal analysis, 
                       Analytical chemistry 29 (11) (1957) 1702–1706.
        """

        logB       = self.logB
        E_KAS      = []
        E_KAS_err  = []
        TempIsoDF  = self.TempIsoDF
        print(f'Kissinger-Akahira-Sunose method: Computing activation energies...')       
        for i in range(TempIsoDF.shape[0]):  
        #Linear regression over all the conversion values in the isoconversional Dataframes   
            y = (logB)- np.log((TempIsoDF.iloc[i].values)**1.92)          #log[1/(T**1.92)]
            x = 1/(TempIsoDF.iloc[i].values)                              #1/T
            LR = linregress(x,y) 
            E_a_i = -(self.R)*(LR.slope)                                  #Activation energy
            error = -(self.R)*(LR.stderr)                                 #Standard deviation of the activation energy
            E_KAS_err.append(error)
            E_KAS.append(E_a_i)

        E_KAS   = np.array(E_KAS)
        KAS_s = np.array(E_KAS_err)
        #Tuple with the results: Activation energy, Standard deviation
        self.E_KAS   = (E_KAS, KAS_s) 
        print(f'Done.')
        return self.E_KAS  
#-----------------------------------------------------------------------------------------------------------
    def I_Temp(self, E, row_i, col_i, method):
        """
        Temperature integral for the Vyazovkin method: \int_{T0}^{T} exp[E_{alpha}/RT]dT

        Parameters:         E        :  Activation energy value in kJ/mol to compute the integral
 
                            row_i    :  DataFrame index value associated to the conversion value of 
                                        the computation.
  
                            col_i    :  DataFrame column associated to the heating rate of the computation

                            method   :  Method to compute the integral temperature. The available methods 
                                        are: 'senum-yang' for the Senum-Yang approximation, 'trapezoid' for
                                        the the trapezoid rule of quadrature, 'simpson' for the simpson rule
                                        and 'quad' for using a technique from the Fortran library QUADPACK 
                                        implemented in the scipy.integrate subpackage.

        Returns:            Float. Result of the division of the integral value by the heating rate. 
                                  
        """
        
        TempIsoDF = self.TempIsoDF
        Beta      = self.Beta 
        #Heating rate for thee computation
        B  = Beta[col_i]              
        #Initial experimental temperature. Lower limit in the temperature integral
        T0 = self.T0[col_i]
        #Upper limit in the temperature integral
        T  = TempIsoDF[TempIsoDF.columns[col_i]][TempIsoDF.index.values[row_i]]
        #Value of the Arrhenius exponential for the temperature T0 and the energy E
        y0 = np.exp(-E/(self.R*(T0)))
        #Value of the Arrhenius exponential for the temperature T and the energy E
        y  = np.exp(-E/(self.R*(T)))
        #Senum-Yang approximation
        def senum_yang(E):
            x = E/(self.R*T)
            num = (x**3) + (18*(x**2)) + (88*x) + (96)
            den = (x**4) + (20*(x**3)) + (120*(x**2)) +(240*x) +(120)
            s_y = ((np.exp(-x))/x)*(num/den)
            return (E/self.R)*s_y

        if method == 'trapezoid':
            I = integrate.trapezoid(y=[y0,y],x=[T0,T])
            #Division of the integral by the heating rate to get the factor $I(E,T)/B$
            I_B = I/B                                   
            return I_B
       
        elif method == 'senum-yang':
            I = senum_yang(E)
            #Division of the integral by the heating rate to get the factor $I(E,T)/B$
            I_B = I/B
            return I_B
        
        elif method == 'simpson':
            I = integrate.simpson(y=[y0,y],x=[T0,T])
            #Division of the integral by the heating rate to get the factor $I(E,T)/B$
            I_B = I/B
            return I_B

        elif method == 'quad':
            def Temp_int(T,E):
                return np.exp(-E/(self. R*(T)))

            I = integrate.quad(Temp_int,T0,T,args=(E))[0]
            #Division of the integral by the heating rate to get the factor $I(E,T)/B$
            I_B = I/B
            return I_B
        else:
            raise ValueError('method not recognized')

#-----------------------------------------------------------------------------------------------------------
    def omega(self,E,row,method):
        """
        Calculates the function to minimize for the Vyazovkin method:

        \Omega(Ea) = \sum_{i}^{n}\sum_{j}^{n-1}{[B_{j}{I(E,T_{i})]}/[B_{i}{I(E,T_{j})}]}        

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row    : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 temperatures.         

                        method : Method to compute the integral temperature.
                                 The available methods are: 'senum-yang' for
                                 the Senum-Yang approximation, 'trapezoid' for
                                 the the trapezoid rule of numerical integration,
                                 'simpson' for the simpson ruleand 'quad' for using 
                                 a technique from the Fortran library QUADPACK 
                                 implemented in the scipy.integrate subpackage.

        Returns:        O      : Float. Value of the omega function for the given E.  
        """ 
        Beta    = self.Beta
        omega_i = []
        method = method
        #Array from a comprehension list of factors of \Omega(Ea)
        p = np.array([self.I_Temp(E,row,i, method=method) for i in range(len(Beta))])
        #Double sum
        for j in range(len(Beta)):
            y = p[j]*((np.sum(1/(p)))-(1/p[j]))
            omega_i.append(y)
        return np.sum((omega_i))
   
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
                               'simpson' for the simpson ruleand 'quad' for using a technique 
                               from the Fortran library QUADPACK implemented in the scipy.integrate
                               subpackage.

        Returns:      A matplotlib figure plotting omega vs E. 
        """
        #Temperature DataFrame
        IsoDF   = self.TempIsoDF
        #Quadrature method
        method = method
        #Activation energy (independent variable) array
        E = np.linspace(bounds[0], bounds[1], N)
        #Evaluation of \Omega(E)
        O = np.array([float(self.omega(E[i],row,method)) for i in range(len(E))])
        #Plot settings
        plt.style.use('seaborn-whitegrid')
        plt.plot(E,O,color='teal',label=r'$\alpha$ = '+str(np.round(IsoDF.index[row],decimals=3)))        
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
        plt.legend()
        plt.grid(True)

        return plt.show()

#-----------------------------------------------------------------------------------------------------------        
    def variance_Vy(self, E,row_i, method):

        """
        Calculates the variance of the activation energy E obtained with the Vyazovkin 
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
                                 'simpson' for the simpson rule and 'quad' for using
                                 a technique from the Fortran library QUADPACK 
                                 implemented in the scipy.integrate subpackage.

        Returns:        Float object. Value of the variance associated to a given E.  

        --------------------------------------------------------------------------------------------
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """ 
        #Heating rates array
        Beta      = self.Beta
        #Temperature Dataframes
        TempIsoDF = self.TempIsoDF
        #Total number of addends
        N = len(Beta)*(len(Beta)-1)
        #Temperature integrals into a list comprehrension
        I = np.array([self.I_Temp(E, row_i, i, method) for i in range(len(Beta))])
        #Each value to be compared with one (s-1) to compute the variance     
        s = np.array([I[i]/I for i in range(len(I))])

        return np.sum((s-1)**2)/N
#-----------------------------------------------------------------------------------------------------------
    def psi_Vy(self, E, row_i, method):
        """
        Calculates the F distribution to minimize for the Vyazovkin method.
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
                                 'simpson' for the simpson rule and 'quad' for 
                                 using a technique from the Fortran library QUADPACK 
                                 implemented in the scipy.integrate subpackage.

        Returns:        error  : Float. Value of the error calculated for a 95% confidence.  
        --------------------------------------------------------------------------------------------
        
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """ 
        Beta      = self.Beta
        TempIsoDF = self.TempIsoDF
        #F values for a 95% confidence interval for (n-1) and (n-1) degreees of freedom
        F      = [161.4, 19.00, 9.277, 6.388, 5.050, 4.284, 3.787, 3.438, 3.179,2.978,2.687] 
        #F value for the n-1 degrees of freedom.
        #Subtracts 1 to n (len(B)) because of degrees of freedom and 1 because of python indexation
        f      = F[len(Beta)-1-1] 
        #quadrature method from parameter "method"
        method = method
        #Psi evaluation interval
        E_p    = np.linspace(1,E+50,50)  
        #'True' value of the activation energy in kJ/mol for a given conversion (row_i)
        E_min  = E          
        #Variance of the 'True' activation energy              
        s_min  = self.variance_Vy(E_min, row_i, method)   
        #Variance of the activation energy array E_p 
        s      = np.array([self.variance_Vy(E_p[i], row_i, method) for i in range(len(E_p))])

        #Psi function moved towards negative values (f-1) in order 
        #to set the confidence limits such that \psy = 0 for those values
        Psy_to_cero = (s/s_min)-f-1      
        
        #Interpolation function of \Psy vs E to find the roots
        #which are the confidence limits
        inter_func = interp1d(E_p,
                              Psy_to_cero, 
                              kind='cubic',  
                              bounds_error=False, 
                              fill_value="extrapolate")
        #Finding the confidence limits
        zeros = np.array([fsolve(inter_func, E-150)[0],                 
                          fsolve(inter_func, E+150)[0]])
        error = np.mean(np.array([abs(E-zeros[0]), abs(E-zeros[1])]))

        return error         
      
#-----------------------------------------------------------------------------------------------------------
    def error_Vy(self,E, method):
        """
        Method to calculate the distribution to minimize for the Vyazovkin method.

        Parameters:     bounds   : Tuple object containing the lower and upper limit values 
                                   for E, to evaluate omega.

                        method   : Method to compute the integral temperature. The available 
                                   methods are: 'senum-yang' for the Senum-Yang approximation,
                                   'trapezoid' for the the trapezoid rule of numerical integration,
                                   'simpson' for the Simpson rule and 'quad' for using a technique 
                                   from the Fortran library QUADPACK implemented in the scipy.integrate 
                                   subpackage.

        Returns:        error_Vy : Array of error values associated to the array of activation 
                                   energies obtained by the Vyazovkin method.  
        """         

        error_Vy = np.array([self.psi_Vy(E[i], i,  method) for i in range(len(E))])

        return error_Vy  
#-----------------------------------------------------------------------------------------------------------
    def Vy(self, bounds, method='senum-yang'):
        """
        Method to compute the Activation Energy based on the Vyazovkin treatment.
        \Omega(E_{\alpha})= min[ sum_{i}^{n}\sum_{j}^{n-1}[J(E,T_{i})]/[J(E,T_{j})] ]

        Parameters:   bounds : Tuple object containing the lower and upper limit values 
                               for E, to evaluate omega.

                      method : Method to evaluate the temperature integral. The available 
                               methods are: 'senum-yang' for the Senum-Yang approximation,
                               'trapezoid' for the the trapezoid rule of numerical integration,
                               'simpson' for the Simpson rule and 'quad' for using a technique 
                               from the Fortran library QUADPACK implemented in the scipy.integrate 
                               subpackage.

        Returns :      Tuple of arrays:
                       E_Vy    : numpy array containing the activation energy values 
                                 obtained by the first Vyazovkin method.
             
                       error   : numpy array containing the error associated to the activation energy 
                                 within a 95% confidence interval.
        ------------------------------------------------------------------------------------------------
        Reference:     S. Vyazovkin, D. Dollimor e, Linear and nonlinear procedures in isoconversional 
                       computations of the activation energy of nonisothermal reactions in solids, Journal 
                       of Chemical Information and Computer Sciences 36 (1) (1996) 42–45.
        """
        E_Vy       = []
        Beta       = self.Beta 
        IsoDF      = self.TempIsoDF
        print(f'Vyazovkin method: Computing activation energies...')    

        for k in range(len(IsoDF.index)):
            E_Vy.append(minimize_scalar(self.omega, args=(k,method),bounds=bounds, method = 'bounded').x)

        E_Vy = np.array(E_Vy)

        error     = self.error_Vy(E_Vy,method)
        
        self.E_Vy = (E_Vy, error) 
        print(f'Done.')
        return self.E_Vy     
   
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
        #Computation of the intagral defined in terms of the exponential integral
        #calculated with scipy.special
        J = a*(sp.expi(-a/c)-sp.expi(-a/b)) + c*np.exp(-a/c) - b*np.exp(-a/b)

        return J
#-----------------------------------------------------------------------------------------------------------        
    def J_time(self, E, row_i, col_i, method = 'trapezoid'):
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

                     method  : Numerical integration method. Can be 'trapezoid', 'simpson' or 'quad'.
                               The method corresponds to those implemented in the scipy.integrate
                               subpackage.

        Returns:     J_t     : Float. Value of the integral obtained by a numerical integration method. 
        """    
        timeAdvIsoDF   = self.timeAdvIsoDF
        #Heating rate for the computation
        B  = self.Beta[col_i]
        #Initial experimental temperature
        T0 = self.T0[col_i]
        #Time values associated to the lower limit of the
        #Temperature range set with DataExtraction.Conversion
        t0 = timeAdvIsoDF[timeAdvIsoDF.columns[col_i]][timeAdvIsoDF.index.values[row_i]]
        #Time associated to the i-th conversion value
        t  = timeAdvIsoDF[timeAdvIsoDF.columns[col_i]][timeAdvIsoDF.index.values[row_i+1]]
        #Value for the Arrhenius exponential for the time t0 and energy E
        y0 = np.exp(-E/(self.R*(T0+(B*t0))))
        #Value for the Arrhenius exponential for the time t and energy E
        y  = np.exp(-E/(self.R*(T0+(B*t))))
            
        if method == 'trapezoid':
            J_t = integrate.trapezoid(y=[y0,y],x=[t0,t])
            return J_t
            
        elif method == 'simpson':
            J_t = integrate.simpson(y=[y0,y],x=[t0,t])
            return J_t
            
        elif method == 'quad':
            def time_int(t,T0,B,E):
                return np.exp(-E/(self.R*(T0+(B*t))))
            
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
        #Array from a comprehension list of factors of \Omega(Ea)
        #The variable of integration depends on the parameter var
        if var == 'Temperature':
            I_x = np.array([self.J_Temp(E,
                                        TempAdvIsoDF[TempAdvIsoDF.columns[i]][TempAdvIsoDF.index[j]],
                                        TempAdvIsoDF[TempAdvIsoDF.columns[i]][TempAdvIsoDF.index[j+1]]) 
                            for i in range(len(TempAdvIsoDF.columns))])
            #Dividing by beta to get the factor $I(E,T)/B$
            I_B = I_x/Beta
            #Double sum
            omega_i = np.array([I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k])) for k in range(len(Beta))])
            O = np.array(np.sum((omega_i)))
            return O
  
        elif var == 'time':
            I_B = np.array([self.J_time(E,
                                        row,
                                        i,
                                        method) 
                            for i in range(len(timeAdvIsoDF.columns))])
            #Double sum
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
        #Temperature DataFrame
        TempAdvIsoDF = self.TempAdvIsoDF
        #time DataFrame
        timeAdvIsoDF = self.timeAdvIsoDF
        #Heating Rates
        Beta         = self.Beta
        #Activation energy (independent variable) array
        E = np.linspace(bounds[0], bounds[1], N)
        #Evaluation of \Omega(E)
        O = np.array([float(self.adv_omega(E[i],row,var,method)) for i in range(len(E))])
        plt.style.use('seaborn-whitegrid')
        plt.plot(E,O,color='teal',label=r'$\alpha$ = '+str(np.round(timeAdvIsoDF.index[row],decimals=3)))
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
        plt.legend()
        plt.grid(True)

        return plt.show()
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
        #Total number of addends
        N = len(self.Beta)*(len(self.Beta)-1)
        #Selection of the integral based on parameter "var"
        if var == 'time':
            #lower limit 
            inf = self.timeAdvIsoDF.index.values[row_i] 
            #upper limit
            sup = self.timeAdvIsoDF.index.values[row_i+1]
            #initial temperature
            T0  = self.T0
            #time integrals into a list comprehension        
            J = np.array([self.J_time(E, row_i, i, method) for i in range(len(self.Beta))])     
            #Each value to be compared with one (s-1) to compute the variance
            s = np.array([J[i]/J for i in range(len(J))])
            
            return np.sum((s-1)**2)/N
            
        elif var == 'Temperature':
            #lower limit
            inf = self.TempAdvIsoDF.index.values[row_i] 
            #upper limit
            sup = self.TempAdvIsoDF.index.values[row_i+1]
        
            #temperature integrals into a list comprehension 
            J = [self.J_Temp(E, 
                            self.TempAdvIsoDF[self.TempAdvIsoDF.columns[i]][inf], 
                            self.TempAdvIsoDF[self.TempAdvIsoDF.columns[i]][sup]) 
                 for i in range(len(self.Beta))]
            #Each value to be compared with one (s-1) to compute the variance
            s = np.array([J[i]/J for i in range(len(J))])
            
            return np.sum((s-1)**2)/N

        else:
            raise ValueError('variable not valid')

#-----------------------------------------------------------------------------------------------------------        
    def psi_aVy(self, E, row_i, var = 'time', method = 'trapezoid'):
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
        #F values for a 95% confidence interval for (n-1) and (n-1) degreees of freedom
        F      = [161.4, 19.00, 9.277, 6.388, 5.050, 4.284, 3.787, 3.438, 3.179,2.978,2.687]
        #F value for the n-1 degrees of freedom
        #Subtracts 1 to n (len(B)) because of degrees of freedom and 1 because of python indexation
        f      = F[len(self.Beta)-1-1]
        #Quadrature method from parameter "method"              
        method = method
        #Psi evaluation interval
        E_p    = np.linspace(1,E+50,50)  #intervalo para evaluar Psi
        #'True' value of the activation energy in kJ/mol for a given conversion (row_i)
        E_min  = E
        #Variance of the 'True' activation energy 
        s_min  = self.variance_aVy(E_min, row_i,var, method) 
        #Variance of the activation energy array E_p 
        s      = np.array([self.variance_aVy(E_p[i], row_i, var, method) for i in range(len(E_p))])
        
        #Psi function moved towards negative values (f-1) in order 
        #to set the confidence limits such that \psy = 0 for those values
        Psy_to_cero = (s/s_min)-f-1
        
        #Interpolation function of \Psy vs E to find the roots
        #which are the confidence limits
        inter_func = interp1d(E_p,
                              Psy_to_cero, 
                              kind='cubic',  
                              bounds_error=False, 
                              fill_value="extrapolate")
        #Finding the confidence limits
        zeros = np.array([fsolve(inter_func, E-150)[0],
                          fsolve(inter_func, E+150)[0]])
        
        error = np.mean(np.array([abs(E-zeros[0]), abs(E-zeros[1])]))

        return error 
#-----------------------------------------------------------------------------------------------------------            
    def error_aVy(self, E, var = 'time', method = 'trapezoid'):
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
        method = method
        error_aVy = np.array([self.psi_aVy(E[i], i, var=var, method=method) for i in range(len(E))])

        return error_aVy  


#-----------------------------------------------------------------------------------------------------------
    def aVy(self,bounds, var='time', method='trapezoid'):
        """
        Method to compute the Activation Energy based on the Advanced Vyazovkin treatment.
        \Omega(E_{\alpha})= min[ sum_{i}^{n}\sum_{j}^{n-1}[J(E,T_{i})]/[J(E,T_{j})] ]

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
        --------------------------------------------------------------------------------------
        References:   S. Vyazovkin, Evaluation of activation energy of thermally stimulated
                      solid-state reactions under arbitrary variation of temperature, Journal
                      of Computational Chemistry 18 (3) (1997) 393–402.

                      S. Vyazovkin, Modification of the integral isoconversional method to
                      account for variation in the activation energy, Journal of Computational
                      Chemistry 22 (2) (2001) 178–183.
        """

        timeAdvIsoDF = self.timeAdvIsoDF
        Beta         = self.Beta
        print(f'Advanced Vyazovkin method: Computing activation energies...')
        #Computing and minimization of \Omega(E) for the conversion values in the isoconversional DataFrame    
        E_aVy        = [minimize_scalar(self.adv_omega,bounds=bounds,args=(k,var,method), method = 'bounded').x 
                        for k in range(len(timeAdvIsoDF.index)-1)]

        E_aVy   = np.array(E_aVy)

        error   = self.error_aVy(E_aVy, var, method)

        self.E_aVy =  (E_aVy, error)
        print(f'Done.')
        return self.E_aVy
#-----------------------------------------------------------------------------------------------------------
    def T_prom(self,TempIsoDF):
        """
        Computes mean values for temperature at isoconversional values
        in order to evaluate the dependence of the activation energy 
        with temperature

        Parameters:     TempIsoDF   :  Isoconversional DataFrame of Temperatures.

        Returns:        T_prom      :  Array of mean temperatures at isoconversional 
                                       values
        """
        T_prom = []
        for i in range(len(TempIsoDF.index.values)):
            Ti = np.mean(TempIsoDF.iloc[i].values)
            T_prom.append(Ti)
        T_prom = np.array(T_prom)
        
        return T_prom
#-----------------------------------------------------------------------------------------------------------

    def export_Ea(self, E_Fr=False, E_OFW=False, E_KAS=False, E_Vy=False, E_aVy=False, file_t="xlsx" ):
        """
        Method to export activation energy values and their uncertainty calculated as either a csv or xlsx file.
         
        Parameters:    E_Fr     : tuple of activation energies and its uncertainty obtained by de Friedman method.

                       E_OFW    : tuple of activation energies and its uncertainty obtained by de OFW method.

                       E_KAS    : tuple of activation energies and its uncertainty obtained by de KAS method.

                       E_Vy     : tuple of activation energies and its uncertainty obtained by de Vyazovkin method.

                       E_aVy    : tuple of activation energies and its uncertainty obtained by de advanced Vyazovkin
                                method.

                       file_t   :  String. Type of file, can be 'csv' of 'xlsx'.
                                  'xlsx' is the default value.

        returns:       If 'xlsx' is selected, a spreadsheet containg one sheet per experiment
                       containing the values of the activation energies. 
                       If 'csv' is selected, one 'csv' file containing the activation energies. 
       """
        TempIsoDF    = self.TempIsoDF
        TempAdvIsoDF = self.TempAdvIsoDF
        Beta         = self.Beta

        print(f"Exporting activation energies...")
        #Conversion values for the isoconversional evaluations
        alps     = TempIsoDF.index.values
        #Mean values for temperature at isoconversional values
        Temp     = self.T_prom(TempIsoDF)
        
        columns = ['alpha']
        columns.append('Temperature [K]')

        #If the value of a parameter is set to True two columns 
        #are added to the file: Activation energy values and its
        #associated error  
        if E_Fr == True:
            E_Fr = self.E_Fr
            columns.append('Fr [kJ/mol]')
            columns.append('Fr_error [kJ/mol]')
        else:
            pass

        if E_OFW == True:
            E_OFW = self.E_OFW
            columns.append('OFW [kJ/mol]')
            columns.append('OFW_error [kJ/mol]')
        else:
            pass

        if E_KAS == True:
            E_KAS = self.E_KAS
            columns.append('KAS [kJ/mol]')
            columns.append('KAS_error [kJ/mol]')
        else:
            pass

        if E_Vy == True:
            E_Vy = self.E_Vy
            columns.append('Vyazovkin [kJ/mol]')
            columns.append('Vy_error [kJ/mol]')
        else:
            pass

        #pandas.DataFrame to be converted to a xlsx or csv file
        DF_Nrgy = pd.DataFrame([], columns = columns)
        #The first column is conversion
        DF_Nrgy['alpha']  = alps
        #The second column is temperature in Kelvin
        DF_Nrgy['Temperature [K]'] = Temp
        #The next columns depends on which parameters were set to True
        if 'Fr [kJ/mol]' in columns:
            DF_Nrgy['Fr [kJ/mol]']=E_Fr[0]                 #Activation energies in kJ/mol
            DF_Nrgy['Fr_error [kJ/mol]']=E_Fr[1]           #Associated error in kJ/mol
        else:
            pass
        if 'OFW [kJ/mol]' in columns:
            DF_Nrgy['OFW [kJ/mol]']=E_OFW[0]               #Activation energies in kJ/mol
            DF_Nrgy['OFW_error [kJ/mol]']=E_OFW[1]         #Associated error in kJ/mol
        else:
            pass
        if 'KAS [kJ/mol]' in columns:
            DF_Nrgy['KAS [kJ/mol]'] = E_KAS[0]             #Activation energies in kJ/mol
            DF_Nrgy['KAS_error [kJ/mol]']=E_KAS[1]         #Associated error in kJ/mol
        else:
            pass
        if 'Vyazovkin [kJ/mol]' in columns:
            DF_Nrgy['Vyazovkin [kJ/mol]'] = E_Vy[0]        #Activation energies in kJ/mol
            DF_Nrgy['Vy_error [kJ/mol]']=E_Vy[1]           #Associated error in kJ/mol
        else:
            pass

        #The advanced Vyazovkin method has to be exported
        #apart because its index length is differet from the
        #other methods
        if E_aVy == True:
            #pandas.DataFrame to save the advanced Vyazovking method results
            adv_DF = pd.DataFrame()
            #Activation energies
            E_aVy = self.E_aVy
            #DataFrame columns
            ad_col = ['alpha',                       #Conversion
                      'Temperature [K]',             #Temperature
                      'adv.Vyazovkin [kJ/mol]',      #Activation energies in kJ/mol
                      'aVy_error [kJ/mol]']          #Associated error in kJ/mol
            #Conversion values for the isoconversional evaluations
            adv_alps = TempAdvIsoDF.index.values[1:]
            #Mean values for temperature at isoconversional values
            adv_Temp = T_prom(TempAdvIsoDF)
   
            #Filling the columns with thier corresponding values
            adv_DF['alpha'] = adv_alps             
            adv_DF['Temperature [K]'] = adv_Temp[1:]
            adv_DF['adv.Vyazovkin [kJ/mol]'] = E_aVy[0]
            adv_DF['aVy_error [kJ/mol]']=E_aVy[1]

        else:
            pass


        #The format of the file is set with the parameter "file_t"
        if(file_t=='xlsx'):
            #For methods Fr, KAS, OFW and Vy
            name1 = 'Activation_Energies_Results.xlsx'
            #For method aVy
            name2 = 'Advanced_Vyazovkin_Results.xlsx'
            #If any parameter was set to True, do nothing
            if len(columns) == 2:
                pass
            #else, create the corresponding file
            else:
                with pd.ExcelWriter(name1) as writer1:
                    DF_Nrgy.to_excel(writer1, sheet_name='Activation Energies',index=False)   
 
            if E_aVy == True:
                with pd.ExcelWriter(name2) as writer2:
                    adv_DF.to_excel(writer2, sheet_name='Advanced Vyazovkin Method',index=False)
            else:
                pass
            print('Resulst saved as {0} and {1}'.format(name1,name2)) 
   
        elif(file_t=='csv'):
            #For methods Fr, KAS, OFW and Vy
            name1 ='Activation_Energies_Results.csv'
            #For method aVy
            name2 = 'Advanced Vyazovkin Method.csv'
            #If any parameter was set to True, do nothing
            if len(columns) == 2:
                pass
            #else, create the corresponding file
            else:
                DF_Nrgy.to_csv(name1, 
                               encoding='utf8', 
                               sep=',',
                               index=False)
            if E_aVy == True:
                adv_DF.to_csv(name2, 
                               encoding='utf8', 
                               sep=',',
                               index=False)
            else:
                pass
            print('Resulst saved as {0} and {1}'.format(name1,name2)) 
        else:
            raise ValueError("File type not recognized")

        print(f'Done.')        
#-----------------------------------------------------------------------------------------------------------        
    def prediction(self, E = None, B = 1, T0 = 298.15, Tf=1298.15):
        """
        Experimental method. May raise error or give unreliable results.





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


