# pICNIK 

pICNIK is a package for isoconversional computations for non-isothermal kinetcis.\
The package has an object oriented interface with two classes: DataExtraction and ActivationEnergy, with the purpose of managing the experimental data and computing activation energies (![formula](https://render.githubusercontent.com/render/math?math=E_{\alpha})) with the next isoconversional methods: 

- Ozawa-Flynn-Wall (OFW)\
![formula](https://render.githubusercontent.com/render/math?math=\ln{\left(\beta_{i}\right)} = \left[\ln{\left(\frac{A_{\alpha}E_{\alpha}}{g(\alpha)R}\right)}-5.331\right]-1.052\frac{E_{\alpha}}{RT_{\alpha,i}}}) 

- Kissinger-Akahira-Sunose (KAS)\ 
$\ln{\left(\frac{\beta_{i}}{T_{\alpha ,i}^{2}}\right)}\approx\ln{\left[\frac{A_\alpha R}{E_\alpha g(\alpha)}\right]}-\frac{E_\alpha}{RT_{\alpha ,i}}$

- Friedman (Fr)\
$\ln{\left(\frac{d\alpha}{dt}\right)_{\alpha ,i}} = \ln{\left[A_{\alpha}f\left(\alpha\right)\right]} - \frac{E_{\alpha}}{RT_{\alpha ,i}}$

- Vyazovkin (Vy)\
$\phi=n(n-1) - \sum_{i}^{n} \sum_{j \neq i}^{n-1} \frac{\beta_j I(E_{\alpha},T_{\alpha ,i})}{\beta_i I(E_{\alpha},T_{\alpha ,j})}$

- Advanced method of Vyazovkin (aVy)\
$\phi=n(n-1) - \sum_{i}^{n} \sum_{j \neq i}^{n-1} \frac{\beta_j J(E_{\Delta\alpha},T_{\Delta\alpha ,i})}{\beta_i J(E_{\Delta\alpha},T_{\Delta\alpha ,j})}$

The repository consist in the following directories:
- pyace.py. Contains the package
- examples. Contains a script (example.py) which executes some commmands of pyace in order to ilustrate the suggested prcedure. And three more directories which contain data to use whith example.py:
    - Constant_E. Simulated TGA data for a process with constant activation energy.
    - Two_Steps. Simulated TGA data for a process with two steps, each with constant activation energy.
    - Variable_E. Simulated TGA data for a process with variable activation energy.


### Installation

`picnik` can be installed from PyPi with `pip`:
`$ pip install picnik`


### DataExtractioin class

It has methods to open the .csv files containing the thermogravimetric data as pandas DataFrames for the experimental data, computing and adding the conversion for the process ($\alpha$) and the conversion rate ($d\alpha/dt$) as columns in the DataFrame.\
The class also has methods for creating isoconversional DataFrames of time, temperature, conversion rates (for the OFW, KAS, Fr and Vy methods) and also "advanced" DataFrames of time and temperature (for the aVy method).\
Example:

    import picnik as pnk
 
    files = ["HR_1.csv","HR_2.csv",...,"HR_n.csv"]
    xtr = pnk.DataExtraction()
    xtr.set_data(files)
    xtr.data_extraction()
    xtr.isoconversional()
    xtr.adv_isoconversional()
    
The DataFrames are stored as attributes of the `xtr` object 


### ActivationEnergy class

This class has methods to compute the activation energies with the DataFrames created with the `xtr` object along with its associated error. The `Fr()`,`OFW()`,`KAS()` methods return a tuple of three, two and two elements respectively. The first element of the tuples is a numpy array containing the isoconversional activation energies. The second element contains the associated error within a 95\% confidence interval. The third element in the case of the `Fr()` method is a numpy array containing the intercept of the Friedman method. The `Vy()` and `aVy()` only return a numpy array of isoconversional activation energies, the error associated to this methods are obtained with the `Vy_error()` and `aVy_error()` methods
Example:

    ace = pnk.ActivationEnergy(xtr.Beta,
                               xtr.TempIsoDF,
                               xtr.diffIsoDF,
                               xtr.TempAdvIsoDF,
                               xtr.timeAdvIsoDF)
    E_Fr, E_OFW, E_KAS, E_Vy, E_aVy = ace.Fr(), ace.OFW(), ace.KAS(), ace.Vy(), ace.aVy()
    Vy_e, aVy_e = ace.Vy_error(), ace.aVy_error()
    
The constructor of this class needs five arguments, a list/array/tuple of Temperature rates, and four DataFrames: one of temperature, one of convertsion rates and two "advanced" one of temperature and the other of time.

### Saving results

The DataExtractionclass also has a method to export the results as .csv or .xlsx files:

    xtr.save_df(E_Fr = E_Fr[0], 
                E_OFW = E_OFW[0], 
                E_KAS = E_KAS[0], 
                E_Vy = E_Vy, 
                E_aVy = E_aVy,
                file_t="xlsx" )
