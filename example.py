#!/usr/bin/python3

import sys
import clases as c

if __name__=='__main__':
    files = sys.argv[1:] # the files in beta ascendent order
    print(files)
    
    #First,
    #Data extraction and preprocessing is made
    #with DataExtraction class as follows:
    extractor = c.DataExtraction()
    extractor.set_datos(files) #Needs the file list as parameter
    extractor.data_extraction()
    extractor.isoconversional()
    extractor.adv_isoconversional()

    #processed data 
    df_iso = extractor.get_df_isoconv()
    df_adv = extractor.get_adviso()
    da_dt = extractor.get_dadt()
    T = extractor.get_T()
    t = extractor.get_t()
    Beta = extractor.Beta
    #Or the hole set of data can 
    #be gotten with
    [df_iso, df_adv, da_dt,T,t,Beta] = extractor.get_valores()


    #Then, 
    #Activation energy
    #Computations are made by the
    #ActivationEnergy class
    acten = c.ActivationEnergy(df_iso, Beta, df_adv) 
    acten.FOW()
    acten.KAS()
    bounds = acten.set_bounds((0,300)) # user can delimit bounds for evaluating E with Vy and Adv_Vy methods
    print("The bounds for evaluating E are "+str(bounds))
    acten.visualize_omega(0,N=1000) # this function allows to evaluate if the bounds are relevant
    acten.vy((1,300))
    acten.vyz((1,300))
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
