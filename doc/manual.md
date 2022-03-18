## Quick guide into pICNIK

The files containing the thermogravimetric (TG) data must be in comma-separeted value (CSV) format having three columns:**time**, **temperature in celsius degrees**, and **mass**.
First thing to do is create a list of file-paths with the TG data as previously described.

    files = ['Usr/filepath_1','Usr/filepath_2',...,'Usr/filepath_n',]

Now, enters `pICNIK` and the `DataExtraction` class with its methods to organize the raw data with the following commands:

    #import the module
    import picnik as pnk

    #instanciate DataExtraction object
    xtr = pnk.DataExtraction()
    #read file (this method has an optional "encoding" parameter)
    Beta, T0 = xtr.read_files(files)
    #compute files for a given temperature range
    xtr.Conversion(T0,Tf)
    #create isoconversional pandas DataFrames
    TDF,tDF,dDF,TaDF,taDF = xtr.Isoconversion(advanced=True,
                                              method='points',
                                              N = len(xtr.TempIsoDF))
   
Next, the `ActivationEnergy` class takes the organized data to compute activation energy:

    #instantiate ActivationEnergy object
    ace = pnk.ActivationEnergy(Beta, #or xtr.Beta
                               T0, #or xtr.T0
                               TDF, #or xtr.TempIsoDF
                               dDF, #or xtr.diffIsoDF
                               TaDF, #or xtr.TempAdvIsoDF
                               taDF, #or xtr.timeAdvIsoDF)
    #Compute activation energies with their associated errors
    E_Fr = ace.Fr()      #for the Friedman method
    E_OFW = ace.OFW()    #for the Ozawa-Flynn-Wall method
    E_KAS = ace.KAS()    #for the Kissinger-Akahira-Sunose method
    E_Vy = ace.Vy(bounds=(1,300),
                  method='senum-yang)      #for the Vyazovkin method
    E_aVy = ace.aVy(bounds=(1,300),
                    var='tme',
                    method='trapezoid')      #for the advanced Vyazovkin method

Finally to export results the `export_Ea()` method is used

    #export results as xlsx 
    ace.export_Ea(E_Fr = True,
                  E_OFW = True,
                  E_KAS = True,
                  E_Vy = True,
                  E_aVy = True,
                  file_t= "xlsx" )
  
