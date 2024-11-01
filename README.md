# pICNIK 

pICNIK is a module with implemented isoconversional computations for non-isothermal kinetcis.\
The package has an object oriented interface with two classes:`DataExtraction` and `ActivationEnergy`, with the purpose of managing the experimental data and computing activation energies  with the next isoconversional methods: 

- Ozawa-Flynn-Wall (OFW)\
- Kissinger-Akahira-Sunose (KAS)\ 
- Friedman (Fr)\
- Vyazovkin (Vy)\
- Advanced method of Vyazovkin (aVy)\

Additionally, the `ActivationEnergy` class contains methods to make isothermal and non-isothermal predictions based on the isoconversional principle. Furthermore, the class has methods to compute the pre-exponential factor by means of the compensation effect and to reconstruct the reaction model in its integrl expression ($g(\alpha)$) given an activation energy ($E$) and an pre-exponential factor ($A$).

The repository consist in the following directories:
- picnik.py. Contains the package
- examples. Contains a script (example.py) which executes some commmands of picnik in order to ilustrate the suggested procedure. And three more directories which contain data to use with example.py:
    - Constant_E. Simulated TGA data for a process with constant activation energy.
    - Two_Steps. Simulated TGA data for a process with two steps, each with constant activation energy.
    - Variable_E. Simulated TGA data for a process with variable activation energy.
- picnik_Test.ipynb. A jupyter notebook test some of the implemented functions.
- pICNIK_walk-through.ipynb. A jupyter notebook with a simple one-step process example and a guide to the usage of the library.
- poetry.lock. Defines the version of the dependencies for picnik.

### Installation

`picnik` can be installed from PyPi with `pip`:
`$ pip install picnik`


### `DataExtraction` class

It has methods to open the .csv files containing the thermogravimetric data as pandas DataFrames for the experimental data, computing and adding the conversion for the process and the conversion rate as columns in the DataFrame.\
The class also has methods for creating isoconversional DataFrames of time, temperature, conversion rates (for the OFW, KAS, Fr and Vy methods) and also "advanced" DataFrames of time and temperature (for the aVy method).\
Example:

    from picnik import picnik as pnk
 
    files = ["HR_1.csv","HR_2.csv",...,"HR_n.csv"]
    xtr = pnk.DataExtraction()
    Beta, T0 = xtr.read_files(files,encoding)
    xtr.Conversion(T0,Tf)
    IsoTables = xtr.Isoconversion(advanced=(bool))
    
    
The DataFrames are also stored as attributes of the `xtr` object 


### ActivationEnergy class

This class has methods to compute the activation energies with the DataFrames created with the `xtr` object along with its associated error. The `Fr()`,`OFW()`,`KAS()` methods return a tuple of three, two and two elements respectively. The first element of the tuples is a numpy array containing the isoconversional activation energies. The second element contains the associated error within a 95\% confidence interval. The third element in the case of the `Fr()` method is a numpy array containing the intercept of the Friedman method. The `Vy()` and `aVy()` only return a numpy array of isoconversional activation energies, the error associated to this methods are obtained with the `Vy_error()` and `aVy_error()` methods
Example:

    ace = pnk.ActivationEnergy(Beta,
                               T0,
                               IsoTables)
    E_Fr, E_OFW, E_KAS, E_Vy, E_aVy = ace.Fr(), ace.OFW(), ace.KAS(), ace.Vy((E_min,E_max)), ace.aVy((E_min,E_max))
    
The constructor of this class needs six arguments, a list/array/tuple of Temperature rates, a list/array of initial temperatures and four DataFrames: one of temperature, one of convertsion rates and two "advanced" one of temperature and the other of time.

#### Pre-exponential factor

 The pre-exponential factor is computed by means of the so-called compensation effect, which implies a linear relation between the pre-exponential factor and the activation energy: $\ln{A}=a+bE$

 A linear regression is computed over a set of {$E_{i}$,$\ln{A_{i}}$} to obtain the parameters $a$ and $b$.
 The values of {$E_{i}$,$\ln{A_{i}}$} are obatined from fitting different models $f(\alpha)_{i}$ (defined in the picnik.rxn_models submodule) to the experimental data

 All this information is returned from the `ActivationEnergy.compensation_effect` method

    ln_A,a, b, Avals, Evals = ace.compensation_effect(E_aVy,B=ace.Beta[0])

#### Model reconstruction

 The numerical reconstruction of the reaction model is carried on in its integral form, $g(\alpha)$
 Given an array of activation energy, $E$, and an array of pre-exponential factor, the integral reaction model can be computed as: 
 $g(\alpha) = \sum_{i} g(\alpha_{i}) = \sum_{i} A_{\alpha_{i}} \int_{t_{\alpha_{i-1}}}^{t_{\alpha_{i}}}\exp(-\frac{E_{\alpha_{i}}}{RT(t_{\alpha_{i}})})dt$

    g_r  = ace.reconstruction(E_aVy,np.exp(ln_A), 3)

#### Isothermal prediction
The `ActivationEnergy`class contains three methods for isothermal prediction, each based on a different equation:\

 a) Model based prediction:          $t_{\alpha_{i}} = \frac{\sum_{i}g(\alpha_{i})}{A\exp{(-\frac{E}{RT_{0}})}}$   ...(1)\
 b) Isoconversion prediction A:      $t_{\alpha_{i}} = \frac{\int_{t_{\alpha_{0}}}^{t_{\alpha_{i}}}\exp(-\frac{E}{RT(t)})}{\exp{(-\frac{E}{RT_{0}})}}$   ...(2)\
 c) Isoconversion prediction B:      $J[E_{\alpha},T(t)]=J[E_{\alpha},T_{0}]$   ...(3)\

 As it can be seen from the expressions above, the methods do not compute conversion as a funciton of time, but they compute the time required to reach a given conversion

    tim_pred1 = ace.t_isothermal(E_aVy,np.exp(ln_A),Tiso,col=0,g_a=g_r,alpha=alpha)        # eq (1)
    tim_pred2 = ace.t_isothermal(E_aVy,np.exp(ln_A),Tiso,col=0,isoconv=True)               # eq (2) 
    ap,Tp,tp  = ace.modelfree_prediction(E_aVy,B=0,isoT =575, alpha=0.999, bounds=(10,10)) # eq (3)

#### Non-isothermal prediction

The `ActivationEnergy.prediction(` method is so general that it can be used to simulate conversion under an arbitrary temperature program which may be a linear or user-defined as a python function.

    ap2,Tp2,tp2 = ace.modelfree_prediction(E_aVy,B=10,alpha=0.999,bounds=(10,10))   # linear heating rate of 10 K/min    


    def Temp_program(t):
	"""
	Temperaure program with isothermal and linear steps: a linear ramp with heating rate of 5 K/min for
	53 minutes (to reach a temperature of 575 K), then, isothermal until process reaches a 100% conversion
	"""
	if t <= 53:
	    return 35+273 + 5*t
	else:
	    return 575

    ap3,Tp3,tp3 = ace.modelfree_prediction(E_aVy,B=0,T_func = Temp_program,alpha=0.999,bounds=(10,10))

### Exporting results

The `ActivationEnergy`class also has methods to export the results as .csv or .xlsx files:

    ace.export_Ea()

    ace.export_prediction(time, Temp, alpha, name="prediction.csv")

    ace.export_kinetic_triplet(E, ln_A, g_a, name="kinetic_triplet.csv")

