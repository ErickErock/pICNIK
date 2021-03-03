#!/usr/bin/python3

import sys
import clases as c

if __name__=='__main__':
    archivos = sys.argv[1:] # the files in beta ascendent order
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
    [df_iso, df_adv, da_dt,T,t,Beta] = get_valores()


    #Then, 
    #Activation energy
    #Computations are made by the
    #ActivationEnergy class
    acten = c.ActivationEnergy(df_iso, beta, df_adv) 
    acten.FOW()
    acten.KAS()
    bounds = (0,300) # user can delimit bounds for evaluating E with Vy and Adv_Vy methods
    visualize_omega(alpha,N) # this function allows to evaluate if the bounds are relevant
    acten.vy(bounds)
    acten.vyz(bounds)

    #Activation Energies
    E_FOW = acten.get_E_FOW()
    E_KAS = acten.get_E_KAS()
    E_Vy  = acten.get_Evy()
    E_AdVy= acten.get_E_vyz()

    #Saving Data

