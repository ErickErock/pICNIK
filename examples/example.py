#!/usr/bin/python3

import sys
import picnik as pnk

if __name__=='__main__':
    files = sys.argv[1:] # the files in beta ascendent order
    print(files)
    
    #instanciate DataExtraction object
    xtr = pnk.DataExtraction()
    #read file (this method has an optional "encoding" parameter)
    Beta, T0 = xtr.read_files(files)
    #compute files for a given temperature range
    xtr.Conversion(323,1023)
    #create isoconversional pandas DataFrames
    TDF,tDF,dDF,TaDF,taDF = xtr.Isoconversion(advanced=True,
                                              method='points',
                                              N = len(xtr.TempIsoDF))
   

    #instantiate ActivationEnergy object
    ace = pnk.ActivationEnergy(Beta, #or xtr.Beta
                               T0, #or xtr.T0
                               TDF, #or xtr.TempIsoDF
                               dDF, #or xtr.diffIsoDF
                               TaDF, #or xtr.TempAdvIsoDF
                               taDF) #or xtr.timeAdvIsoDF
    #Compute activation energies with their associated errors
    E_Fr = ace.Fr()      #for the Friedman method
    E_OFW = ace.OFW()    #for the Ozawa-Flynn-Wall method
    E_KAS = ace.KAS()    #for the Kissinger-Akahira-Sunose method
    E_Vy = ace.Vy(bounds=(1,300),
                  method='senum-yang')      #for the Vyazovkin method
    E_aVy = ace.aVy(bounds=(1,300),
                    var='tme',
                    method='trapezoid')      #for the advanced Vyazovkin method


    #export results as xlsx 
    ace.export_Ea(E_Fr = True,
                  E_OFW = True,
                  E_KAS = True,
                  E_Vy = True,
                  E_aVy = True,
                  file_t= "xlsx" )
  
