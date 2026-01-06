#!/usr/bin/python3.7
"""
This module has two classes: DataExtraction and ActivationEnergy. 
DataExtraction reads csv files and creates pandas.DataFrames according to the isoconversional principle. 
ActivationEnergy computes the activation energy (E) with five implemented isoconversional methods: 
Friedman (Fr), Ozawa-Flynn-Wall(OFW), Kissinger-Akahira-Sunos (KAS) and the method developed by Vyazovkin (Vy, aVy).
Additionally, ActivationEnergy possesses methods to reconstruct de reaction model in its integral form, g(a),
methods to compute the compensation effect parameters (a,b) and the pre-exponential factor (A), as well as methods
to make model-free or model-based predictions.
"""
from typing import Any

#Dependencies
import numpy as np                                                      # (1) numpy
import pandas as pd                                                     # (2) pandas
from scipy.signal import savgol_filter
from   scipy.interpolate import interp1d, make_smoothing_spline         # (3) scipy
from   scipy.optimize    import minimize_scalar, fsolve, curve_fit      
import scipy.special     as     sp
from scipy.stats import linregress, f                                   
from scipy import integrate
import seaborn as sns                                        # colors for the plots
import matplotlib.pyplot as plt                                         # (4) matplotlib
plt.rcParams.update({'font.size': 16})

from picnik_integrator import picnik_integrator as integ                # (*) 
from rxn_models import rxn_models                                       # (*)


"""
For information about the depencies we refer the user to:
    (1) https://numpy.org/doc/stable/
        Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
    (2) https://pandas.pydata.org/
        Data structures for statistical computing in python, McKinney, Proceedings of the 9th Python in Science Conference, Volume 445, 2010.
    (3) https://docs.scipy.org/doc/scipy/
        Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
    (4) https://matplotlib.org/
        J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
    (*) This module come along with the picnik package. https://github.com/ErickErock/pICNIK/ 
"""
#-----------------------------------------------------------------------------------------------------------
class DataExtraction:
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
        self.BetaError      = []              #list of heating rate associated error
        self.files          = []              #list of files containing raw data
        self.da_dt          = []              #list of experimental conversion rates 
        self.T              = []              #list of experimental temperature in Kelvin
        self.T0             = []              #list of experimental initial temperature in Kelvin
        self.t              = []              #list of experimental time
        self.alpha          = []              #list of experimental conversion
        self.TempAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional temperature DataFrame
        self.timeAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional time DataFrame
        self.diffAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional conversion rate DataFrame
#-----------------------------------------------------------------------------------------------------------    
    def read_files(self, flist, encoding='utf8', diff_smoother = 'SG', summary=True):
        """ 
        Reads each TGA file as a pandas DataFrame and calculates de heating rate. Each DataFrame is stored
        as an element of the attribute 'DFlis'. 
        The columns of each DataFrame are: 'time [min]', 'Temperature [C]', 'mass [mg]', '%m', 'Temperature [K]', 
        
        Parameters:    flist : list object containing the paths of the files to be used.
        
                       encoding : The available encodings for pandas.read_csv() method. Includes but not limited 
                                  to 'utf8', 'utf16','latin1'. For more information on the python standar encoding:
                                  (https://docs.python.org/3/library/codecs.html#standard-encodings).

                       diff_smoother: String. Method to smooth the numerical derivative: Available options are 'SG'
                                      for a Savitzky-Golay filter with a window on i% the lenght of the array and
                                      cubic polynomial. Or 'Sp3' for a B cubic spline with smoothing parameter lambda=0.5.

                       summary: Bool. True for a graphic summary of the data.

        Returns:       Beta: Array of the fitted heating rates.
                       
                       T0: Array of experimental initial temperatures.
        """
    
        print("Files to be used: \n{}\n ".format(flist))
        DFlis    =   self.DFlis
        Beta     =   self.Beta
        BetaEr   =   self.BetaError
        T0       =   self.T0
        print(f'Reading files and creating DataFrames...\n')
        for item in flist:
            #csv files can use a tab or a coma as separator.
            try:
                DF = pd.read_csv(item,  sep = '\t', encoding = encoding)
                #stores the initial temperature of the ith experiment
                T0.append(DF[DF.columns[1]][0]+273.15)                       
            except IndexError:
                DF = pd.read_csv(item,  sep = ',', encoding = encoding)
                #stores the initial temperature of the ith experiment
                T0.append(DF[DF.columns[1]][0]+273.15)                       
            #computes the mass loss percentage
            DF['%m'] = 100*(DF[DF.columns[2]]/DF[DF.columns[2]][0])
            #creates a column for the temperature in Kelvin
            DF['Temperature [K]'] = DF[DF.columns[1]] + 273.15      
            #Derivatives with smoothing filters
            dwdt = np.gradient(DF[DF.columns[2]].values,
                       DF[DF.columns[0]].values,
                               edge_order=2)
            dwdt_p = np.gradient(DF['%m'].values,
                                 DF[DF.columns[0]].values,
                                 edge_order=2)
            dTdt = np.gradient(DF['Temperature [K]'].values,
                               DF[DF.columns[0]].values)
            if diff_smoother == 'SG':
                try:
                    dwdt_sm = savgol_filter(dwdt,
                                            int(len(dwdt)*0.01),
                                            3,
                                            mode='nearest')
                    dwdt_p_sm = savgol_filter(dwdt_p,
                                              int(len(dwdt_p) * 0.01),
                                              3,
                                              mode='nearest')
                    dTdt_sm = savgol_filter(dTdt,
                                            int(len(dTdt) * 0.01),
                                            3,
                                            mode='nearest')
                except ValueError:
                    dwdt_sm = savgol_filter(dwdt,
                                            int(len(dwdt) * 0.1),
                                            3,
                                            mode='nearest')
                    dwdt_p_sm = savgol_filter(dwdt_p,
                                              int(len(dwdt_p) * 0.1),
                                              3,
                                              mode='nearest')
                    dTdt_sm = savgol_filter(dTdt,
                                            int(len(dTdt) * 0.1),
                                            3,
                                            mode='nearest')
            elif diff_smoother == 'Sp3':
                spl_dwdt = make_smoothing_spline(DF[DF.columns[0]].values,
                                            dwdt,
                                            lam=0.5)
                dwdt_sm = spl_dwdt(DF[DF.columns[0]])

                spl_dwdt_p = make_smoothing_spline(DF[DF.columns[0]].values,
                                            dwdt_p,
                                            lam=0.5)
                dwdt_p_sm = spl_dwdt_p(DF[DF.columns[0]])

                spl_dTdt = make_smoothing_spline(DF[DF.columns[0]].values,
                                            dTdt,
                                            lam=0.5)
                dTdt_sm = spl_dTdt(DF[DF.columns[0]])
            else:
                raise ValueError("Smoothing method not available. Type 'SG' for a Savitxky-Golay filter or 'Sp3' for a B-cubic spline.")
            DF['dw/dt'] = DF[DF.columns[0]]
            DF['dw/dt'] = dwdt_sm
            #computes the differential thermogram with a Savitzki-Golay filter                                    
            dwdt_p = np.gradient(DF['%m'].values,
                                 DF[DF.columns[0]].values)
            DF['dw/dt [%/min]'] = DF[DF.columns[0]]
            DF['dw/dt [%/min]'] = dwdt_p_sm
            #computes the heating rate with a Savitzki-Golay filter


            DF['dT/dt'] = DF[DF.columns[0]]
            DF['dT/dt'] = dTdt_sm
            
            #computes the heating rate
            LR = linregress(DF[DF.columns[0]],                            
                            DF[DF.columns[1]])

            BetaEr.append(LR.intercept_stderr)
            Beta.append(LR.slope)
            DFlis.append(DF)

        self.DFlis     = DFlis                     #List of the DataFrames constructed
        self.Beta      = np.array(Beta)            #Array of heating rates in ascendent order
        self.BetaError = np.array(BetaEr)   #Array of correlation coefficients for the heating rates
        self.T0        = np.array(T0)              #Array of experimental initial temperatures

        print(f'The computed heating rates are:\n')
        for b in range(len(Beta)):
            print(f'\n{Beta[b]:6.3f} +/- {BetaEr[b]:.3f} K/min\n')

        if summary == True:
            #TG visualization
            fig, axs = plt.subplots(2 , 3,figsize=(18,8))
            fig.suptitle('Summary of the input data',fontsize=20)
            fig.tight_layout()
            # mass% vs time
            for i in range(len(DFlis)):
                axs[0][0].plot(DFlis[i][DFlis[0].columns[0]],
                               DFlis[i]['%m'],
                               lw=3,
                               label=rf'$\beta$={Beta[i]:.1f} K/min')
                axs[0][0].legend(fontsize=14)
                axs[0][0].set_xlabel('time [min]')
                axs[0][0].set_ylabel('mass [%]')
            # mass loss rate vs time
            for i in range(len(DFlis)):
                axs[0][1].plot(DFlis[i][DFlis[0].columns[0]],
                               DFlis[i]['dw/dt [%/min]'],
                               lw=3,
                               label=rf'$\beta$={Beta[i]:.1f} K/min')
                axs[0][1].legend(fontsize=14)
                axs[0][1].set_xlabel('time [min]')
                axs[0][1].set_ylabel('dw/dt [%/min]')
            # heating rate vs time
            for i in range(len(DFlis)):
                axs[0][2].plot(DFlis[i][DFlis[0].columns[0]],
                               DFlis[i]['dT/dt'],
                               lw=3,
                               label=rf'$\beta$={Beta[i]:.1f} K/min')
                axs[0][2].legend(fontsize=14)
                axs[0][2].set_xlabel('time [min]')
                axs[0][2].set_ylabel('dT/dt [K/min]')
            # mass% vs temperature
            for i in range(len(DFlis)):
                axs[1][0].plot(DFlis[i]['Temperature [K]'],
                               DFlis[i]['%m'],
                               lw=3,
                               label=rf'$\beta$={Beta[i]:.1f} K/min')
                axs[1][0].legend(fontsize=14)
                axs[1][0].set_xlabel('Temperature [K]')
                axs[1][0].set_ylabel('mass [%]')
            # mass loss rate vs temperature
            for i in range(len(DFlis)):
                axs[1][1].plot(DFlis[i]['Temperature [K]'],
                               DFlis[i]['dw/dt [%/min]'],
                               lw=3,
                               label=rf'$\beta$={Beta[i]:.1f} K/min')
                axs[1][1].legend(fontsize=14)
                axs[1][1].set_xlabel('Temperature [K]')
                axs[1][1].set_ylabel('dw/dt [%/min]')
            # heating rate vs temperature
            for i in range(len(DFlis)):
                axs[1][2].plot(DFlis[i]['Temperature [K]'],
                               DFlis[i]['dT/dt'],
                               lw=3,
                               label=rf'$\beta$={Beta[i]:.1f} K/min')
                axs[1][2].legend(fontsize=14)
                axs[1][2].set_xlabel('Temperature [K]')
                axs[1][2].set_ylabel('dT/dt [K/min]')
            plt.show()
        else: pass
        return self.Beta, self.T0 
#-----------------------------------------------------------------------------------------------------------
    def plot_data(self,x_data='time',y_data='TG',x_units='min',y_units='%'):
        """
        Visualizer of the input data. 
        Plots available: mass vs time/temperature ; mass loss rate vs time/temperature; heating rate vs time/temperature.

        Parameters:     x_data: String. Data to plot in the 'x' axis. Options are: 'time'(default) or 'temperature'.
                        
                        y_data: String. Data to plot in the 'y' axis. Options are: 'TG'(default) for the thermogram, 
                                        'DTG' for the differential thermogram or 'dT/dt' for the heating rate.

                        x_units: String. Units of the x-data. Options depend on the x_data parameter. If
                                 x_data='time', the only value for this parameter is: 'min'(default). If x_data='temperature', options 
                                 are: 'C' or 'K'.

                        y_units: String. Units of the y-data. Options depend on the y_data parameter:
                                 For 'TG', options are: '%'(default) or 'mg'.
                                 For 'DTG', options are: '%/min', 'mg/min', '%/s' or 'mg/s'.
                                 For 'dT/dt', options are: 'K/min'(default), 'C/min', 'K/s' or 'C/s'.
        Returns:        A plot of x_data vs y_data.
                     
        """
        #Data containers
        DFlis = self.DFlis
        Beta  = self.Beta
        #x-data and x-units
        x_dat = ['time','temperature']
        x_un  = ['min','K', 'C']
        #y-data and y-units
        y_dat = ['TG','DTG','dT/dt']
        y_un  = ['%','mg','%/min','mg/min','K/min', 'C/min']
        #All combinations of x-data+x-units and y-data+y-units         
        x_keys = [dx+ux for dx in x_dat for ux in x_un]
        y_keys = [dy+uy for dy in y_dat for uy in y_un]

        #Dictionaries of valid combinations of data and units.
        #Keys are formed from the function parameters. 
        #Values correspond to the column index in the DataFrames of the atrribute DFlis.
        #Valid combinations of x-data+x-units
        x_dict={x_keys[0]:0,
                x_keys[4]:4,
                x_keys[5]:1}
        #Valid combinations of y-data+y-units
        y_dict={y_keys[0]:3,
                y_keys[1]:2,
                y_keys[8]:5,
                y_keys[9]:6,
                y_keys[16]:7,
                y_keys[17]:7}

        plt.figure(figsize=(15,7))
        k = x_dict[x_data+x_units]
        l = y_dict[y_data+y_units]
        for i in range(len(DFlis)):
            plt.plot(DFlis[i][DFlis[i].columns[k]],
                     DFlis[i][DFlis[i].columns[l]],
                     lw=3,
                     label=r'$\beta$ = '+str(np.round(Beta[i],decimals=1)))

        plt.xlabel(x_data+' ['+x_units+']')
        plt.ylabel(y_data+' ['+y_units+']')
        plt.legend()
        plt.show()
#-----------------------------------------------------------------------------------------------------------
    # noinspection PyUnboundLocalVariable
    def Conversion(self,T0, Tf, diff_smoother ='SG'):
        """
        Calculates the conversion values for a given temperature range. 
        Not all experimental points are suitable for the isoconversional 
        analysis, so a temperature analysis range must be selected based 
        on the thermal profile of the sample.
        
        Parameters:    T0: List of initial temperatures in Kelvin of the interval where the process to study is.
                         
                       Tf: List of final temperatures in Kelvin of the interval where the process to study is.

                       diff_smoother: String. Method to smooth the numerical derivative: Available options are 'SG'
                                      for a Savitzky-Golay filter with a window of 1% or 10% the lenght of the array
                                      and cubic polynomial. Or 'Sp3' for a B cubic spline with smoothing parameter 
                                      lambda=0.5.

        Returns:       A plot of the temperature range to be used in the analysis.
                       
        """
        DFlist            = self.DFlis
        NDFl              = []
        print(f'Computing conversion values...')
        for i in range(len(DFlist)):
            #filters the DataFrames based on the temperature limits
            item = DFlist[i]
            item = item.loc[(item['Temperature [K]'] > T0[i]) & (item['Temperature [K]'] < Tf[i])]
            item = item.reset_index(drop=True)
            #calculates the conversion
            item['alpha'] = (item[item.columns[2]][0]-item[item.columns[2]])/(item[item.columns[2]][0]-item[item.columns[2]][item.shape[0]-1])
            #computes the cnversion rate with a Savitzki-Golay filter
            dadt = np.gradient(item['alpha'].values,
                               item[item.columns[0]].values)
            if diff_smoother == 'SG':
                try:
                    dadt_sm = savgol_filter(dadt,
                                            int(len(dadt) * 0.01),
                                            3,
                                            mode='nearest')
                except ValueError:
                    dadt_sm = savgol_filter(dadt,
                                            int(len(dadt) * 0.1),
                                            3,
                                            mode='nearest')
            elif diff_smoother == 'Sp3':
                spl = make_smoothing_spline(item[item.columns[0]].values,
                                            dadt,
                                            lam=0.5)
                dadt_sm = spl(item[item.columns[0]].values)
            item['da/dt'] = item[item.columns[0]]
            item['da/dt'] = dadt_sm

            item = item.loc[(item['alpha'] > 0.002) & (item['alpha'] < 0.998)]
            NDFl.append(item)

        alpha = []
        T = []
        t = []
        da_dt = []

        #To create the Isoconversional DataFrames interpolation is needed. 
        #In order to make the interpolation the x values must be strictly in ascending order.
        #The next block of code evaluates if the i-th value is bigger than the i-1-th, if so, 
        #the value is appended to the corresponding list. 
        for i in range(len(NDFl)):
            #The initial values are those of the lower limit of the temperature range.
            a = [NDFl[i]['alpha'].values[0]]
            Temp = [NDFl[i]['Temperature [K]'].values[0]]
            time = [NDFl[i][NDFl[i].columns[0]].values[0]]
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

        markers = ["o","v","s","*","x","^","p","<","2",">"]
        #Plot of the thermograms showing the anaysis range.
        fig, ax1 = plt.subplots(figsize=(12,9))

        for i in range(len(NDFl)):
            ax1.plot(NDFl[i]['Temperature [K]'].values[::2],           #Temperature in Kelvin
                     NDFl[i]['alpha'].values[::2],                        #mass loss percentage
                     marker = markers[i],
                     markersize=10,
                     linestyle = '--',
                     linewidth=4.20,
                     label=rf'$\beta=$ {self.Beta[i]:.2f} K/min',
                     alpha=0.75)
        ax1.set_ylabel(r'conversion ($\alpha$)')
        ax1.set_xlabel('Temperature [K]')
        ax1.set_xlim((T0[0]-20),(Tf[-1]+20))
        ax1.legend(frameon=True)
        ax1.grid(True)

        plt.show()
#-----------------------------------------------------------------------------------------------------------
    def Isoconversion(self, d_a = 0.01):    
        """
        Constructs the isoconversional DataFrames.

        Parameters:             d_a:      The step size between the i-th and the i+1-th value in the
                                          conversion array if method is set to 'step'.

        Returns:                3 pandas.DataFrame objects: Temperatures Dataframe, times DataFrame, conversion
                                rates DataFrame.                            
        """
        
        alpha = self.alpha
        T     = self.T
        t     = self.t
        da_dt = self.da_dt
        Beta  = self.Beta
        
        TempAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional temperature DataFrame
        timeAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional time DataFrame
        diffAdvIsoDF   = pd.DataFrame()  #Advanced isoconversional conversion rate DataFrame
       
        print(f'Creating Isoconversion DataFrames...')
        
        adv_alps = np.arange(alpha[-1][0],alpha[-1][-1],d_a)

        for i in range(len(Beta)):
            #New interpolation functions with the advanced conversion array
            inter_func = interp1d(alpha[i], 
                                  T[i],
                                  kind='cubic', 
                                  bounds_error=False, 
                                  fill_value="extrapolate")
            TempAdvIsoDF[rf'$\beta=$ {Beta[i]:.2f} K/min'] = np.round(inter_func(adv_alps), decimals = 4)

            inter_func2 = interp1d(alpha[i], 
                                   t[i],
                                   kind='cubic', 
                                   bounds_error=False, 
                                   fill_value="extrapolate")
            timeAdvIsoDF[rf'$\beta=$ {Beta[i]:.2f} K/min'] = np.round(inter_func2(adv_alps), decimals = 4)
            
            inter_func3 = interp1d(alpha[i], 
                                   da_dt[i],
                                   kind='cubic', 
                                   bounds_error=False, 
                                   fill_value="extrapolate")
            diffAdvIsoDF[rf'$\beta=$ {Beta[i]:.2f} K/min'] = np.round(inter_func3(adv_alps), decimals = 4)
            
            timeAdvIsoDF.index = adv_alps
            TempAdvIsoDF.index = adv_alps
            diffAdvIsoDF.index = adv_alps

            self.TempAdvIsoDF = TempAdvIsoDF      #Isoconversional DataFrame of temperature for the advanced Vyazovkin method (aVy)
            self.timeAdvIsoDF = timeAdvIsoDF      #Isoconversional DataFrame of time for the advanced Vyazovkin method (aVy)
            self.diffAdvIsoDF = diffAdvIsoDF      #Isoconversional DataFrame of conversion rate for the advanced Vyazovkin method (aVy)
            self.d_a          = d_a               #Size of the \Delta\alpha step
        
        print(f'Done')

        return self.TempAdvIsoDF, self.timeAdvIsoDF, self.diffAdvIsoDF
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
    def get_beta_error(self):
        """
        Getter for the correlation coefficient of the heating rates.

        Parameters:   None

        Returns:      list object containing the experimental T vs t correlation coefficient
                      obtained from a linear regression, sorted in correspondance with the 
                      heating rate list (attribute Beta).
        """
        return self.BetaError
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
        return self.TempAdvIsoDF
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
        return self.timeAdvIsoDF
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
        return self.diffAdvIsoDF
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
            plt.ylabel(r'$\text{d}\alpha/\text{d}t [min$^{-1}]$')
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
            plt.ylabel(self.DFlis[i].columns[1])
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
            plt.ylabel(r'$\alpha$')
            plt.legend()
        return plt.show()
#-----------------------------------------------------------------------------------------------------------
class ActivationEnergy:
    """
	Uses the attributes of DataExtraction to compute activation energy values based on five methods: 
    Friedman, FOW, KAS, Vyazovkin and Advanced Vyazovkin.
    This class also contains methods to: compute the pre-exponential factor based on the compensation effect, numerically reconstruct the integral reaction model, compute model-based or model-free predictions and export the aforementioned data.
    """
    def __init__(self, Beta, T0, IsoTables):
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
        self.TempAdvIsoDF = IsoTables[0]     #Isoconversional DataFrame of temperatures
        self.timeAdvIsoDF = IsoTables[1]     #Isoconversional DataFrame of time
        self.diffAdvIsoDF = IsoTables[2]     #Isoconversional DataFrames of conversion rates
        self.T0           = T0               #Array of initial experimental temperatures
        self.E_Fr         = [[],[],[],[]]       #Container for the Friedmann (Fr) method results
        self.E_OFW        = [[],[],[],[]]       #Container for the OFW method (OFW) results
        self.E_KAS        = [[],[],[],[]]       #Container for the KAS method (KAS) results
        self.E_Vy         = [[],[],[],[]]       #Container for the Vyazovkin method (Vy) results
        self.E_aVy        = [[],[],[],[]]       #Container for the advanced Vyazovkin method (aVy)results

        self.R            = 0.0083144626     #Universal gas constant 0.0083144626 kJ/(mol*K)
        self.used_methods = []
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
        a_Fr      = []
        E_Fr      = []
        E_Fr_err  = []
        T_Fr      = []
        Fr_b      = []
        diffIsoDF = self.diffAdvIsoDF
        TempIsoDF = self.TempAdvIsoDF
        T_prom    = TempIsoDF.mean(axis=1).values
        print(f'Friedman method: Computing activation energies...')
        for i in range(0,diffIsoDF.shape[0]):
            try:
            #Linear regression over all the conversion values in the isoconversional Dataframes
                y     = np.log(diffIsoDF.iloc[i].values)             #log(da_dt)
                x     = 1/(TempIsoDF.iloc[i].values)                 #1/T
                LR    = linregress(x,y)
                E_a_i = -(self.R)*(LR.slope)                         #Activation Energy

                a_Fr.append(TempIsoDF.index.values[i])
                E_Fr.append(E_a_i)            
                T_Fr.append(T_prom[i])
                Fr_b.append(LR.intercept)                            #ln[Af(a)]
                error = -(self.R)*(LR.stderr)                        #Standard deviation of the activation energy
                E_Fr_err.append(error)
            except ValueError:
                pass

        a_Fr   = np.array(a_Fr)
        E_Fr   = np.array(E_Fr)
        T_Fr   = np.array(T_Fr)
        Fr_e   = np.array(E_Fr_err)
        Fr_b   = np.array(Fr_b)
        #Tuple with the results: Conversion, Temperature, Activation energy, Standard deviation and ln[Af(a)]
        self.E_Fr =  (a_Fr, T_Fr, E_Fr, np.abs(Fr_e), Fr_b)
        self.used_methods.append('Fr')
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
        a_OFW      = []
        T_OFW      = []
        E_OFW      = []
        E_OFW_err  = []
        TempIsoDF  = self.TempAdvIsoDF
        T_prom     = TempIsoDF.mean(axis=1).values
        print(f'Ozawa-Flynn-Wall method: Computing activation energies...')        
        for i in range(TempIsoDF.shape[0]):
            try:
            #Linear regression over all the conversion values in the isoconversional Dataframes
                y = (logB)                                           #log(\beta)
                x = 1/(TempIsoDF.iloc[i].values)                     #1/T
                LR = linregress(x,y)
                E_a_i = -(self.R/1.052)*(LR.slope)                   #Activation energy
                error = -(self.R/1.052)*(LR.stderr)                  #Standard deviation of the activation energy
                E_OFW_err.append(error)
                T_OFW.append(T_prom[i])
                E_OFW.append(E_a_i)
                a_OFW.append(TempIsoDF.index.values[i])
            except ValueError:
                pass

        a_OFW = np.array(a_OFW)
        E_OFW = np.array(E_OFW)
        T_OFW = np.array(T_OFW)
        OFW_s = np.array(E_OFW_err)   
        #Tuple with the results: Conversion, Temperature, Activation energy, Standard deviation
        self.E_OFW   = (a_OFW,T_OFW, E_OFW, np.abs(OFW_s))
        self.used_methods.append('OFW')
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
        a_KAS      = []     
        T_KAS      = []
        E_KAS      = []
        E_KAS_err  = []
        TempIsoDF  = self.TempAdvIsoDF
        T_prom     = TempIsoDF.mean(axis=1).values
        print(f'Kissinger-Akahira-Sunose method: Computing activation energies...')       
        for i in range(TempIsoDF.shape[0]):
            try:
            #Linear regression over all the conversion values in the isoconversional Dataframes   
                y = (logB)- np.log((TempIsoDF.iloc[i].values)**1.92)          #log[1/(T**1.92)]
                x = 1/(TempIsoDF.iloc[i].values)                              #1/T
                LR = linregress(x,y) 
                E_a_i = -(self.R)*(LR.slope)                                  #Activation energy
                error = -(self.R)*(LR.stderr)                                 #Standard deviation of the activation energy
                a_KAS.append(TempIsoDF.index.values[i])
                T_KAS.append(T_prom[i])
                E_KAS_err.append(error)
                E_KAS.append(E_a_i)
            except ValueError:
                pass
        
        a_KAS  = np.array(a_KAS)
        E_KAS  = np.array(E_KAS)
        T_KAS  = np.array(T_KAS)
        KAS_s  = np.array(E_KAS_err)
        #Tuple with the results: Conversion, Temperature, Activation energy, Standard deviation
        self.E_KAS   = (a_KAS, T_KAS, E_KAS, np.abs(KAS_s)) 
        self.used_methods.append('KAS')
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
        
        TempIsoDF = self.TempAdvIsoDF
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
        IsoDF   = self.TempAdvIsoDF
        #Quadrature method
        method = method
        #Activation energy (independent variable) array
        E = np.linspace(bounds[0], bounds[1], N)
        #Evaluation of \Omega(E)
        O = np.array([float(self.omega(E[i],row,method)) for i in range(len(E))])
        #Plot settings
        plt.style.use('seaborn-v0_8-whitegrid')
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
        TempIsoDF = self.TempAdvIsoDF
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
        TempIsoDF = self.TempAdvIsoDF
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
        a_Vy       = []
        T_Vy       = []
        E_Vy       = []
        Beta       = self.Beta 
        IsoDF      = self.TempAdvIsoDF
        DF_prom    = IsoDF.mean(axis=1).values
        print(f'Vyazovkin method: Computing activation energies...')    

        for k in range(len(IsoDF.index)):
            E_Vy.append(minimize_scalar(self.omega, args=(k,method),bounds=bounds, method = 'bounded').x)
            a_Vy.append(IsoDF.index.values[k])
            T_Vy.append(DF_prom[k])

        E_Vy = np.array(E_Vy)
        T_Vy = np.array(T_Vy)
        a_Vy = np.array(a_Vy)
        error     = self.error_Vy(E_Vy,method)
        
        self.E_Vy = (a_Vy, T_Vy, E_Vy, error) 
        self.used_methods.append('Vy')
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
    def J_time(self, E, row_i, col_i,ti=None, tf=None, Beta=None, T_func=None, isothermal=False, isoT=0):
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

        Returns:     J_t     : Float. Value of the integral obtained by a numerical integration method. 
        """    

        def time_int(t,*args):
            T0 = args[0]
            B  = args[1]
            E  = args[2]
            if T_func is None:
                T  = T0+(B*t)   
            else:
                T  = T_func(np.array([t]))[0]
            return np.exp(-E/(self.R*(T)))
       
        if tf is None:
            timeAdvIsoDF   = self.timeAdvIsoDF
            #Heating rate for the computation
            B  = self.Beta[col_i]
            #Initial experimental temperature
            T0 = self.T0[col_i]
            t0 = timeAdvIsoDF[timeAdvIsoDF.columns[col_i]][timeAdvIsoDF.index.values[row_i]]
            t  = timeAdvIsoDF[timeAdvIsoDF.columns[col_i]][timeAdvIsoDF.index.values[row_i+1]]
        else:
            t0 = ti
            t  = tf
            if isothermal is False:
                T0 = np.mean(self.T0)
                B  = Beta
            else:
                T0 = isoT
                B = 0
        J_t = integ.Trapezoid(time_int, t0, t, 1, T0, B, E)[0]
        return J_t
#-----------------------------------------------------------------------------------------------------------        
    def adv_omega(self,E, row, var = 'time'):
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

        Returns:      O       : Float. Value of the advanced omega function for a given E.
        """
        TempAdvIsoDF = self.TempAdvIsoDF
        timeAdvIsoDF = self.timeAdvIsoDF
        Beta         = self.Beta
        j            = row
        n = len(Beta)*(len(Beta)-1)
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
                                        i) 
                            for i in range(len(timeAdvIsoDF.columns))])
            #Double sum
            #omega_i = np.array([I_B[k]*((np.sum(1/(I_B)))-(1/I_B[k])) for k in range(len(Beta))])
            omega_i = np.array([((I_B[k] / I_B) -1)**2 for k in range(len(Beta))])
            O = np.array(np.sum((omega_i)))/n
            return O        
#-----------------------------------------------------------------------------------------------------------
    def visualize_advomega(self,row,var='time',bounds=(1,300),n=1000):
        """
        Method to visualize adv_omega function. 

        Parameters:   row     : Index value for the row of conversion in the pandas DataFrame
                                containing the isoconversional times or temperatures.
                     
                      var     : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'. Default 'time'.
      
                      bounds  : Tuple object containing the lower limit and the upper limit values 
                                of E, for evaluating adv_omega. Default (1,300).

                      N       : Int. Number of points in the E array for the plot. Default 1000.


        Returns:      A matplotlib plot of adv_omega vs E 
        """
        #Temperature DataFrame
        TempAdvIsoDF = self.TempAdvIsoDF
        #time DataFrame
        timeAdvIsoDF = self.timeAdvIsoDF
        #Heating Rates
        Beta         = self.Beta
        #Activation energy (independent variable) array
        E = np.linspace(bounds[0], bounds[1], n)
        #Evaluation of \Omega(E)
        O = np.array([float(self.adv_omega(E[i],row,var)) for i in range(len(E))])
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.plot(E,O,color='teal',label=r'$\alpha$ = '+str(np.round(timeAdvIsoDF.index[row],decimals=3)))
        plt.ylabel(r'$\Omega\left(E_{\alpha}\right)$')
        plt.xlabel(r'$E_{\alpha}$')
        plt.legend()
        plt.grid(True)

        return plt.show()
#-----------------------------------------------------------------------------------------------------------        
    def variance_aVy(self, E, row_i, var = 'time'):
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

        Returns:        Float object. Value of the variance associated to a given E.  

        --------------------------------------------------------------------------------------------
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """
        #Total number of addends
        n = len(self.Beta)*(len(self.Beta)-1)
        #Selection of the integral based on parameter "var"
        if var == 'time':
            #lower limit 
            inf = self.timeAdvIsoDF.index.values[row_i] 
            #upper limit
            sup = self.timeAdvIsoDF.index.values[row_i+1]
            #initial temperature
            T0  = self.T0
            #time integrals into a list comprehension        
            J = np.array([self.J_time(E, row_i, i) for i in range(len(self.Beta))])     
            #Each value to be compared with one (s-1) to compute the variance
            #s = np.array([J[i]/J for i in range(len(J))])
            #return np.sum((s-1)**2
            s = np.array([((J[i] / J)-1)**2 for i in range(len(J))])
            return np.sum(s)/n
            
        elif var == 'Temperature':
            #lower limit
            inf = self.TempAdvIsoDF.index.values[row_i] 
            #upper limit
            sup = self.TempAdvIsoDF.index.values[row_i+1]
        
            #temperature integrals into a list comprehension 
            J: list[float | Any] = [self.J_Temp(E,
                                                self.TempAdvIsoDF[self.TempAdvIsoDF.columns[i]][inf],
                                                self.TempAdvIsoDF[self.TempAdvIsoDF.columns[i]][sup])
                                     for i in range(len(self.Beta))]
            #Each value to be compared with one (s-1) to compute the variance
            s = np.array([J[i]/np.array(J) for i in range(len(J))])
            return np.sum((s-1)**2)/n

        else:
            raise ValueError('Variable not valid.')

#-----------------------------------------------------------------------------------------------------------        
    def psi_aVy(self, E, row_i, var = 'time', p =0.95):
        """
        Method to calculate the F distribution to minimize for the Vyazovkin method.
        The distribution is computed as: 

        \Psi(E) = S^{2}(E)/S^{2}_{min}

        Parameters:     E      : The activation energy value used to calculate 
                                 the value of omega.

                        row_i  : index value for the row of conversion in the
                                 pandas DataFrame containing the isoconversional
                                 times or temperatures.         

                        bounds : Tuple object containing the lower and upper limit values 
                                 for E, to evaluate the variance.

                        var    : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'

        Returns:        Psi    : Float. Value of the distribution function that sets the lower
                                 and upper confidence limits for E.  
        --------------------------------------------------------------------------------------------
        
        Reference:     Vyazovkin, S., & Wight, C. A. (2000). Estimating realistic confidence intervals 
                       for the activation energy determined from thermoanalytical measurements. 
                       Analytical chemistry, 72(14), 3171-3175.
        """         
        #F values for a p% confidence interval for (n-1) and (n-1) degreees of freedom
        n1, n2 = len(self.Beta)*(len(self.Beta)-1) - 1, len(self.Beta)*(len(self.Beta)-1) - 1
        F = f.ppf(p,n1,n2)
            
        #Psi evaluation interval
        E_p    = np.linspace(1,E+150,150)  #intervalo para evaluar Psi
        #'True' value of the activation energy in kJ/mol for a given conversion (row_i)
        E_min  = E
        #Variance of the 'True' activation energy 
        s_min  = self.variance_aVy(E_min, row_i,var) 
        #Variance of the activation energy array E_p 
        s      = np.array([self.variance_aVy(E_p[i], row_i, var) for i in range(len(E_p))])
        
        #Psi function moved towards negative values (f-1) in order 
        #to set the confidence limits such that \psy = 0 for those values
        Psy_to_cero = (s/s_min)-F
        
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
        
        error = abs(zeros[1] - zeros[0])/2

        return error, Psy_to_cero, E_p 
#-----------------------------------------------------------------------------------------------------------            
    def error_aVy(self, E, var = 'time', p =0.95):
        """
        Method to calculate the distribution to minimize for the Vyazovkin method.

        Parameters:     bounds   : Tuple object containing the lower and upper limit values 
                                   for E, to evaluate adv_omega.

                        var    : The variable to perform the integral with, it can be either
                                'time' or 'Temperature'

        Returns:        error_aVy : Array of error values associated to the array of activation 
                                    energies obtained by the Vyazovkin method.  
        """ 

        error_aVy = np.array([self.psi_aVy(E[i], i, var=var, p=p)[0] for i in range(len(E))])

        return error_aVy  

#-----------------------------------------------------------------------------------------------------------
    def aVy(self,bounds, var='time', p= 0.95):
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

        Returns:      E_aVy   : numpy array containing the activation energy values
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
        T_prom       = self.TempAdvIsoDF.mean(axis=1).values
        print(f'Advanced Vyazovkin method: Computing activation energies...')
        
        E_aVy   = [minimize_scalar(self.adv_omega,bounds=bounds,args=(m,var), method = 'bounded').x
                    for m in range(len(timeAdvIsoDF.index)-1)]
        error   = self.error_aVy(E_aVy, var=var, p=p)
        error = np.array(error)

        E_aVy   = np.array(E_aVy)
        a_aVy = timeAdvIsoDF.index.values[1::]
        T_aVy = T_prom[1::]
        self.E_aVy =  (a_aVy,T_aVy, E_aVy, error)
        self.used_methods.append('aVy')
        print(f'Done.')
        return self.E_aVy
#-----------------------------------------------------------------------------------------------------------
    def Ea_plot(self, errorbar=True, xlim=(0.05,0.95), ylim=(0,300), saveplot=False, name=None):
        """ method to plot the activation energy vs conversion

        Parameters:     errorbar: Bool. If True, the errorbar is plotted. Only y-values are plotted otherwise.

                        xlim: (float, float). Plotting domain interval. Values should be 0 < float < 1.

                        ylim: (float, float). Plotting Range interval.

                        saveplot: Bool. If True, the plot is saved as the value for 'name' (see below).

                        name: String. Path to save the plot. Example: 'Documents/Data/Figs/Ea_plot.csv'

        Returns:        Plot of conversion (x-values) vs activation energy [kJ/mol] (y-values)
        """
        
        E_dict = {'Fr':self.E_Fr, 
                  'KAS':self.E_KAS,
                  'OFW':self.E_OFW,
                  'Vy':self.E_Vy,
                  'aVy':self.E_aVy}
       
        if errorbar:
            for m in self.used_methods:
                plt.errorbar(E_dict[m][0],
                             E_dict[m][2],
                             E_dict[m][3],
                             marker='.',
                             label=m)
        else:
            for m in self.used_methods:
                plt.plot(E_dict[m][0],
                         E_dict[m][2],
                         marker='.',
                         label=m)


        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$E_{\alpha}$')
        plt.legend()
        plt.grid(True)
        plt.show()
        if saveplot:
            plt.savefig(name)
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
        return TempIsoDF.mean(axis=1).values
#-----------------------------------------------------------------------------------------------------------

    def export_Ea(self):
        """
        Exports Activation Energy data in .csv format. The exported files contains four 
        columns: Conversion, Temperature, Activation Energy and error associated to the
        Activation Energy.
        """

        print(f'Exporting data in csv format...')
        
        E_dict = {'Fr':self.E_Fr,
                  'KAS':self.E_KAS,
                  'OFW':self.E_OFW,
                  'Vy':self.E_Vy,
                  'aVy':self.E_aVy}

        for k in self.used_methods:
                
            data = {'conversion':E_dict[k][0],
                    'Temperature [K]':E_dict[k][1],
                    'E [kJ/mol]':E_dict[k][2],
                    'error [kJ/mol]':E_dict[k][3]}
            df = pd.DataFrame(data)
            df.to_csv(k+'.csv',index=False)
        print(f'Done')
#-----------------------------------------------------------------------------------------------------------
    def j(self,t,Ei,col,row,t_prime,B,isoT=None,T_func=None):
        """
        Auxliliary function to compute model-free predictions
        """
        J0  = self.J_time(Ei, row, col)

        t_func = None
        isot = 0
        iso = False

        if isoT is not None:
            iso =True,
            isot = isoT
        elif T_func is not None:
            t_func = T_func

        Ji = self.J_time(Ei,row, col, ti=t_prime[row],tf=t,Beta=B,
                         T_func=t_func, isothermal=iso, isoT=isot)
        return (Ji-J0)**2
#-----------------------------------------------------------------------------------------------------------

    def calculate_J(self,t,Ei,row,t_prime,B,isoT,T_func):
        """
        Auxliliary function to compute model-free predictions
        """
        J = []
        for b in range(len(self.Beta)):
            J.append(self.j(t,Ei,b,row,t_prime,B,isoT,T_func))
        J = np.sum(np.array(J))
        return J

#-----------------------------------------------------------------------------------------------------------

    def modelfree_prediction(self,E, B, isoT = None, T_init=None,T_func = None, alpha=0, bounds = (5,5)):
        """
        Method to compute a model-free prediction based on the integral isoconversional principle: $g(\alpha)=constant$ which implies 
        an equality between teperature integrals to reach a given conversion: $J[E_alpha,T(t)_i] = J[E_alpha,T(t)_j]$
        where $T(t)_i and T(t)_j$ are two different temperature programs

        Parameters: E       :    numpy array containing the values of activation energy.
                    B       :    float. Heating rate for a linear temperature program; T(t) = T0 +Bt
                    isoT    :    float. Temperature for a constant temperature program (isothermal);
                                 T = isoT = constant
                    T_init  :    float. Initial temperature for the linear temperature program (T0). 
                                 If none is given the value will be an average of
                                 the experimental initial temperatures.
                    T_func  :    function with one parameter t. 
                                 A custom program temperature; T = T(t)
                    alpha   :    float between 0 and 1. 
                                 This parameter determines the starting point in the time
                                 domain for the temperature integral. 
                                 The default value (alpha = 0) sets the initial time to an 
                                 averge of the experimental initial times.
                                 Otherwise the initial time equals zero.
                    bounds  :    a 2 element tuple with the lower and upper bounds.
                                 As the computation of the time required to reach a given
                                 conversion involves a minimization procedure this parameters
                                 sets the bounds in the time domain where the minimum would
                                 be reasonable to be found.

        Returns:    a_prime :    Conversion array
                    T_prime :    Temperature array corresponding to the temperature program which the prediction is made for
                    t_prime :    Predicted time to reach each converssion value in the a_prime array
        -----------------------------------------------------------------------------------------------------------------------------------
        References:         Granado, L., & Sbirrazzuoli, N. (2021). Isoconversional computations for nonisothermal kinetic predictions. Thermochimica Acta, 697, 178859. 

        """
        
        if T_init is None:
            T_init = np.mean(self.T0)
        
        tDF  = self.timeAdvIsoDF


        if alpha != 0:
            tDF = tDF.loc[tDF.index.values <= alpha]
            tj_init = 0
        else: 
            t0_int = interp1d(self.Beta,
                          tDF.iloc[0].values[:len(self.Beta)],
                          fill_value='extrapolate')
            tj_init = t0_int(B)
    
        t_prime = [tj_init]
        print('Beginning simulation at : ',tj_init,' min')
        for i in range(len(tDF.index)-1):
            t_min = minimize_scalar(self.calculate_J,args=(E[i],i,t_prime,B,isoT,T_func),
                                   bounds=((t_prime[i]-bounds[0]),(t_prime[i]+bounds[1])),
                                   method = 'bounded').x
            t_prime.append(t_min)

        print("simulation completed")
        t_prime = np.array(t_prime)
        a_prime = tDF.index.values
        if isoT is not None:
            T_prime = np.ones(len(t_prime))*isoT
        elif T_func is None:
            T_prime = T_init + (B*t_prime)
        else:
            T_prime = T_func(t_prime)
        return a_prime, T_prime, t_prime

#----------------------------------------------------------------

    def compensation_effect(self, col, E=None, errorE=None, f_alpha=None, error_m='mse_NL'):
        """
        Function to compute the pre-exponential factor based on the compensation effect
        which states a linear relation between E and ln(A): ln(A) = a + bE.
        May raise give unreliable results.
        Parameters:   E  : numpy array containing the values of activation energy.
                      B  : float. Value of the heating rate for the dataframe index.
                      f_alpha  : List of extra functions of models to iterate over.
                                 By default the function will only iterate over all the functions
                                 on the "rxn_models.py" file
        Returns:      ln_A: numpy array containing the values of the logaritmic preexponential factor
                      a  : the slope of the relationship between E and A
                      b  : the intercept of the relationship between E and A
                      Afit: the fitted values of the preexponential factor
                      Efit: the fitted values of the activation energy
        """
        Tdf = self.TempAdvIsoDF
        Ddf = self.diffAdvIsoDF
        x = Tdf[Tdf.columns[col]].values
        y = Ddf[Ddf.columns[col]].values
        alpha = Tdf.index.values

        def fit(x, y, f_alpha, alpha, er_m=error_m):

            def filter_fit(funcs=f_alpha, rsq_l=0.95, rsq_u=1.05, mse_lim=0.1):

                def g(xaux, A, E):
                    return A * np.exp(-E / (self.R * xaux)) * f(alpha)

                Afit = []
                Efit = []
                r_sqr = []
                model = []
                for f in funcs:
                    try:
                        # noinspection PyTupleAssignmentBalance
                        popt, pcov = curve_fit(g, x, y)
                    except RuntimeError:
                        pass
                    residuals = y - g(x, *popt)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    if er_m == 'r_NL':
                        if r_squared > rsq_l and r_squared < rsq_u and popt[0] > 0:
                            r_sqr += [r_squared]
                            Afit += [popt[0]]
                            Efit += [popt[1]]
                            model += [f]
                        else:
                            pass
                    elif er_m == 'mse_NL':
                        if ss_res < mse_lim and popt[0] > 0:
                            r_sqr += [ss_res]
                            Afit += [popt[0]]
                            Efit += [popt[1]]
                            model += [f]
                        else:
                            pass
                    elif er_m == 'r_Lin':
                        dep = np.log((1 / f(alpha)) * y)
                        ind = 1 / x
                        df = pd.DataFrame({'x': ind,
                                           'y': dep})
                        df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        df = df.dropna()
                        lr = linregress(df['x'].values, df['y'].values)
                        pen = lr.slope
                        ord = lr.intercept
                        # print( lr.rvalue**2 )
                        if (lr.rvalue ** 2) > rsq_l:
                            Afit += [np.exp(ord)]
                            Efit += [-self.R * pen]
                            model += [f]
                            r_sqr += [lr.rvalue ** 2]
                        else:
                            pass

                return np.array(Afit), np.array(Efit), np.array(r_sqr), model

            fr_l  = np.array([0.9999, 0.999, 0.99, 0.95, 0.85])
            fr_u  = 2 - fr_l
            f_mse = (0.0001, 0.001, 0.005, 0.01, 0.05)

            k = 0
            while k < len(fr_l):
                Afit_t, Efit_t, r_sq_t, mod_t = filter_fit(f_alpha,
                                                           rsq_l=fr_l[k],
                                                           rsq_u=fr_u[k],
                                                           mse_lim=f_mse[k])
                if Efit_t.size == 0 or Afit_t.size == 0:
                    k += 1
                    if k == len(fr_l):
                        print('No model managed to pass the accuracy filters.')
                    else:
                        if er_m == 'r_NL' or er_m == 'r_Lin':
                            print(
                                rf'Accuracy not met with precision of $r^{2}$ = {fr_l[k - 1]}. Lowering precision to $r^{2}$ = {fr_l[k]}')
                        else:
                            print(
                                rf'Accuracy not met with precision of mse = {f_mse[k - 1]}. Lowering precision to mse = {f_mse[k]}')
                else:
                    break

            return Afit_t, Efit_t, r_sq_t, mod_t

        def regression(E, A):
            LR = linregress(E, np.log(A))
            a = LR.slope
            b = LR.intercept
            error_a = LR.stderr
            error_b = LR.intercept_stderr
            return a, error_a, b, error_b

        if E is None:
            print('Activation energy values are necessary to compute pre-exponential factor values')
            return None
        else:
            pass
        if errorE is None:
            print('Activation energy errors are necessary to compute pre-exponential factor errors')
            return None
        else:
            pass
        if f_alpha is None:
            f_a = []
        f_a += filter(callable, list(rxn_models.__dict__.values()))
        Afit, Efit, r_sq, mod = fit(x, y, f_a, alpha, er_m=error_m)
        self.accepted_models = mod
        if Efit.size == 0 or Afit.size == 0:
            print('Compensation effect could not be computed for this data.')
            return None
        else:
            a, errora, b, errorb = regression(Efit, Afit)
            ln_A = (a * E) + b
            errorlnA = np.sqrt((errora ** 2) + ((E * errorb) ** 2) + ((b * errorE) ** 2))

            return ln_A, errorlnA, a, errora, b, errorb, np.array(Afit), np.array(Efit), r_sq, mod

    #---------------------------------------------------------------
    def reconstruction(self,E, A, B):

        """ 
        Method to numericaly reconstruct the reaction model in its integral expression, $g(\alpha)$ 
        The reconstructions is computed as $g(\alpha)=\sum_{i}g(alpha_{i})$

        Parameters:     E :   numpy array containing the values of activation energy.
                        A :   numpy array containing the values of Pre-exponential factor array.
                        B :   Float. Value of the heating rate for the assossiated to the temperatures 
                              to be used for the temperature integral

        Returns:        g :   Numerical values (not an analytical function) of the integral reaction model
                              of the process under study
        """
        
        models = self.accepted_models
        g = []
        AiJsum = 0
        Beta = list(self.Beta)
        col = Beta.index(B)
        for i in range(len(E)-1):
            J   = self.J_time(E[i],i,col,Beta=self.Beta[col])
            AiJsum += A[i]*J
            g += [AiJsum]

        #color = sns.color_palette("rocket",len(models))
        #color = sns.color_palette("crest",len(models))
        color = sns.color_palette("Spectral",len(models))
        #color = sns.color_palette("cubehelix", len(models))
        #color = sns.color_palette("icefire", len(models))

        alpha   = self.timeAdvIsoDF.index.values[1::]

        line_wid = 4.20
        font_siz = 14
        for m in range(len(models)):
            sp1 = str(models[m]).split('at')[0]
            sp2 = sp1.split('<')[1]
            plt.plot(alpha, models[m](alpha, integral=True),
                     ls='--', color=color[m], lw=line_wid, label=sp2)

        try:
            plt.plot(alpha,g,'*-',color='#6963DB',lw=4.20, ms= 8, label='g_r',alpha=0.5)
        except ValueError:    
            plt.plot(alpha[1::],g,'*-',color='#6963DB',lw=4.20, ms= 8, label='g_r',alpha=0.5)
        plt.ylim(0,2)
        plt.xlim(0,1)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$g(\alpha)$')
        plt.grid(1)
        plt.legend(fontsize=12)
        plt.show()
        return np.array(g)

#---------------------------------------------------------------
    def t_isothermal(self, E, ln_A, T0, col, g_a=None, alpha=None, isoconv=False):
        """
        Method to compute isothermal, model-based or model-free, predictions. 
        Note that the ActivationEnergy.modelfree_prediction( method also computes 
        isothermal prediction which may be more accurate.

        Parameters:         E       :   Activation energy array.
                            
                            A       :   Pre-exponential factor array.

                            T0      :   Temperature of the system in Kelvin.

                            col     :   The index for the experimental heating rate, B, assossiated to the temperatures 
                                        to be used for the temperature integral

                            g_a     :   This parameter may be one of two types: The first one is an numpy.array-type object as
                                        the one obtained with the ActivationEnergy.reconstruction method. The second one is 
                                        one of the functions defined in rxn_model.py which may be summoned as picnik.rxn_models.MOD,
                                        where MOD is one of the following codes: A2, A3, A4, D1, D2, D3, F1, P2, P2_3, P3, P4, R2 or R3.

                            alpha   :   Conversion array. 
                                
                            isoconv :   Boolean. If True overrides the g_a options and the method computes the time required to reach a 
                                        given conversion value (from the 'alpha' array) based on the isoconversional principle.

        Returns:            t_p     :   Predicted time required to reach each the conversion values in the 'alpha' array. 
        """
        if type(g_a) == np.ndarray:
            G = interp1d(alpha[:-1:],g_a,fill_value='extrapolate')
            m = G(alpha)
            E0=np.mean(E)
            ln_A0=np.mean(ln_A)
            n = np.exp(ln_A0)*np.exp(-E0/(self.R*T0))
            return m/n

        elif type(g_a) == type(rxn_models.A2):
            m=g_a(alpha,integral=True)
            E0=np.mean(E)
            ln_A0=np.mean(ln_A)
            n = np.exp(ln_A0)*np.exp(-E0/(self.R*T0))
            return m/n

        if isoconv:
            t_p = []
            t_tmp = []
            for i in range(len(E)-1):
                J   = self.J_time(E[i],i,col,Beta=self.Beta[col])
                t_i = J/(np.exp(-E[i]/(self.R*T0)))
                t_tmp.append(t_i)
                tp = np.sum(np.array(t_tmp))
                t_p.append(tp)
            return np.array(t_p)
#---------------------------------------------------------------
    def export_prediction(self, time, Temp, alpha, isothermal=False, name="prediction.csv" ):
        """
        Method to export the kinetic prediction.

        Parameters:     time    : Time array.

                        Temp    : Temperature array or, if the prediction is isothermal just a float with the
                                  Temperature value

                        alpha   : Conversion array.
                         
                        isothermal: Bool. If True the value of Temp is multiplied by a numpy.ones array of
                                    the necessary length.

                        name    : File name in .csv format.

        Returns:    None. A file will be created according to the working path or path specified in `name`.
        """
        if isothermal==True:
            Temp = Temp*np.ones(len(time))
        predDF = pd.DataFrame({'time':time,
                               'Temperature':Temp,
                               'conversion':alpha})
        predDF.to_csv(name,index=False)
#---------------------------------------------------------------
    def export_kinetic_triplet(self, alpha, E, ln_A, g_a, name="kinetic_triplet.csv" ):
        """
        Method to export the kinetic prediction.

        Parameters:     time    : Activation energy array.

                        Temp    : Natural logarithm of pre-exponential factor array.

                        g_a     : Model reaction array.

                        name    : File name in .csv format.

        Returns:    None. A file will be created according to the working path or path specified in `name`.
        """
        kinDF = pd.DataFrame({r'$\alpha$':alpha,
                              'E':E,
                              'ln_A':ln_A,
                              'g(alpha)':g_a})
        kinDF.to_csv(name,index=False)



