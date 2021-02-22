#!/usr/bin/python3.7
import numpy as np
import pandas as pd


class DataExtraction(object):

    def __init__(self):
        """
        constructor. 
        No recibe parametros y solo
        establece variables usadas
        """
        self.DFlis          = []
        self.Beta           = []
        self.BetaPC         = []
        self.files          = [] 

    def set_datos(self, lista_archivos):
        """
        Método para establecer lista de 
        archivos para el extractor.
        """
        self.files = lista_archivos


    def data_extraction(self):
        BetaPearsCoeff = self.BetaPC
        DFlis          = self.DFlis
        Beta           = self.Beta
        filelist       = self.files
        print("Archivos a ocupar: \n{} ".format(filelist))

        for item in filelist:

            DF = pd.read_table(item, engine = 'python', encoding="latin1")
            DF['Temperature [K]'] = DF['Temperature (°C)'] + 273.15
            DF['alpha'] = (DF['Weight (mg)'][0]-DF['Weight (mg)'])/(DF['Weight (mg)'][0]-DF['Weight (mg)'][DF.shape[0]-1])

            y = DF['Temperature [K]']
            x = DF['Time (min)']
            den = np.sum((x-(np.mean(x)))**2)
            num = np.sum((x-(np.mean(x)))*(y-(np.mean(y))))
            r = (num**2)/(np.sum((x-np.mean(x))**2)*(np.sum((y-np.mean(y))**2)))

            BetaPearsCoeff.append(r)
            Beta.append(num/den )
            DFlis.append(DF)
        self.BetaPC = BetaPearsCoeff
        self.DFlis  = DFlis


