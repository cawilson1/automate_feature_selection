# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:23:12 2018

@author: Casey
"""

import pandas as pd
import numpy as np

def readFile(_inputColNums, _outputColNums):
    inputData = pd.read_csv("./datasets/kc_house_data/kc_house_data.csv",
                            sep=',',
                            usecols=_inputColNums,
                            header=0)
    outputData = pd.read_csv("./datasets/kc_house_data/kc_house_data.csv",
                             sep=',',
                             usecols=_outputColNums,
                             header=0)

    tempAllX = np.array(inputData, dtype="float")
    
    #minmax scaling
    for index in range(tempAllX.shape[1]):
        avgOfX = np.average(tempAllX[:, index])

        tempAllX[:, index] = (tempAllX[:, index] - avgOfX)/(np.max(tempAllX[:, index] - np.min(tempAllX[:, index])))
 
    return tempAllX, np.array(outputData, dtype="float"), inputData.head(0).columns.values