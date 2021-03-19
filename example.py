#!/usr/bin/python3

import sys
import clases as c

if __name__=='__main__':
    files = sys.argv[1:] # the files in beta ascendent order
    print(files)
    
    #First,
    #Data extraction and preprocessing is made
    #with DataExtraction class as follows:
    extractor = c.DataExtraction() #Creates the extractor object
    extractor.set_data(files) #Needs the file list as parameter
    extractor.data_extraction() #This method fills the internal variables: Beta, BetaCC and DFlis 
    #The variables generated can be called as:
    beta = extractor.get_beta()
    betaCC = extractor.get_betaCC()
    DFlis = extractor.get_DFlis()
    extractor.isoconversional() #This method fills the internal variables: da_dt, T (for da_dt), t (for da_dt) and Iso_convDF
    #Acces to generated data:
    da_dt = extractor.get_dadt() 
    T = extractor.get_T()
    t = extractor.get_t()
    df_iso = extractor.get_df_isoconv()
    extractor.adv_isoconversional() #This method fills the AdvIsoDF internal variable
    df_adv = extractor.get_adviso()
    #Or the main data can be assigned with:
    [df_iso, df_adv, da_dt,T,t,Beta] = extractor.get_values()


    #Then, Activation energy
    #Computations are made with the
    #ActivationEnergy class
    acten = c.ActivationEnergy(df_iso, Beta, df_adv) 
    E_FOW = acten.FOW() #This method returns E_FOW
    E_KAS = acten.KAS() #This method returns E_KAS
    bounds = acten.set_bounds((1,300)) # user can delimit bounds for evaluating E with Vy and Adv_Vy methods
    print("The bounds for evaluating E are "+str(bounds))
    E, O = acten.visualize_omega(0) # this function allows to evaluate if the bounds are relevant. First, creates the x (E) and y (O) values.
    plt.plot(E,O) #then E vs O can be plotted to visualize the omega function
    plt.show()
    acten.vy(bounds)
    acten.vyz(bounds)
    DeltaAlpha = acten.DeltaAlpha()
    print('The integration interval for the Advanced Vyazovkin is:' + str(DeltaAlpha))

    #Activation Energies
    E_FOW = acten.get_E_FOW()
    E_KAS = acten.get_E_KAS()
    E_Vy  = acten.get_Evy()
    E_AdVy= acten.get_EVyz()

    #Finally,
    #to save tha data generated
    #the save_df module is used
    extractor.save_df(E_FOW,E_KAS,E_Vy,E_AdVy,dialect='csv') # dialect can be either 'xlsx' or 'csv'
    #The output of this method
    #depends on the chosen
    #dialect. If the dialect
    #is 'xlsx'the output will
    #be an excel file with n+1
    #sheets, n containing the 
    #processed data and one
    #containing the Activation 
    #Energies. If the dialect 
    # is 'csv' the out put will 
    #be n+1 csv files. 
