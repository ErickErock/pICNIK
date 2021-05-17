#!/usr/bin/python3

import sys
import clases as pyace

if __name__=='__main__':
    files = sys.argv[1:] # the files in beta ascendent order
    print(files)
    
    #First,
    #Data extraction and preprocessing is made
    #with DataExtraction class as follows:
    xtr = pyace.DataExtraction() #Creates the extractor object
    xtr.set_data(files) #Needs the file list as parameter
    xtr.data_extraction() #This method fills the internal variables: Beta, BetaCC and DFlis 
    #The variables generated can be called as:
    beta = xtr.get_beta()
    betaCC = xtr.get_betaCC()
    DFlis = xtr.get_DFlis()
    xtr.isoconversional() #This method fills the internal variables: da_dt, T (for da_dt), t (for da_dt) and Iso_convDF
    #Acces to generated data:
    da_dt = xtr.get_dadt() 
    T = xtr.get_T()
    t = xtr.get_t()
    TempIsoDF = xtr.get_df_isoconv()
    xtr.adv_isoconversional() #This method fills the AdvIsoDF internal variable
    df_adv = xtr.get_adviso()
    #Or the main data can be assigned with:
    [df_iso, df_adv, da_dt,T,t,Beta] = xtr.get_values()


    #Then, Activation energy
    #Computations are made with the
    #ActivationEnergy class
    ace = pyace.ActivationEnergy(df_iso, Beta, df_adv) 
    E_FOW = ace.FOW() #This method returns E_FOW
    E_KAS = ace.KAS() #This method returns E_KAS
    bounds = ace.set_bounds((1,300)) # user can delimit bounds for evaluating E with Vy and Adv_Vy methods
    print("The bounds for evaluating E are "+str(bounds))
    ace.visualize_omega(0) # this method allows to evaluate if the bounds are relevant for the i-th alpha value in the IsoDF. 
    ace.vy(bounds)
    ace.vyz(bounds)
    DeltaAlpha = ace.DeltaAlpha()
    print('The integration interval for the Advanced Vyazovkin is:' + str(DeltaAlpha))

    #Activation Energies
    E_FOW = ace.get_E_FOW()
    E_KAS = ace.get_E_KAS()
    E_Vy  = ace.get_Evy()
    E_AdVy= ace.get_EVyz()

    #Finally,
    #to save tha data generated
    #the save_df module is used
    xtr.save_df(E_FOW,E_KAS,E_Vy,E_AdVy,dialect='csv') # dialect can be either 'xlsx' or 'csv'
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
