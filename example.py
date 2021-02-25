#!/usr/bin/python3

import sys
import clases as c

if __name__=='__main__':
    archivos = sys.argv[1:]
    print(archivos)
    
    #First
    #Data extraction and preprocessing is made
    #with DataExtraction class as follows
    extractor = c.DataExtraction()
    extractor.set_datos(archivos)
    extractor.data_extraction()
    extractor.isoconversional()
    extractor.adv_isoconversional()

    #processed data 
    df_iso = extractor.get_df_isoconv()
    df_adv = extractor.get_adviso()
    da_dt = extractor.get_dadt()
    t = extractor.get_t()
    beta = extractor.Beta


    #Then
    #Calculations are made by
    #ActivationEnergy class
    acten = c.ActivationEnergy(df_iso, beta, df_adv)
    acten.FOW(); acten.KAS()
    bounds = (0,300) # user can delimit bounds
    acten.vy(bounds)
    acten.vyz(bounds)

