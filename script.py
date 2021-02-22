#!/usr/bin/python3

import clases_extraction as CE 
import sys 

extractor = CE.DataExtraction()


if __name__=='__main__':
    archivos = sys.argv[1:]
    extractor.set_datos( archivos )
    extractor.data_extraction()
    extractor.isoconversional()
    extractor.adv_isoconversional()



    
