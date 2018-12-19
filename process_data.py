# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:23:12 2018

@author: Casey
"""

import pandas as pd
import numpy as np

def readAllFeatures(XFile,yFile):
    
    #f = open("file.txt","a")
    #f.write("\n" + XFile + ' ' + yFile + '\n')
    #f.close
 
    inputData = pd.read_csv(XFile,
                            sep=',',
                            header=0)
    outputData = pd.read_csv(yFile,
                             sep=',',
                             header=0)
    
    tempAllX = np.array(inputData,dtype="float")

        #minmax scaling
    for index in range(tempAllX.shape[1]):
        avgOfX = np.average(tempAllX[:, index])

        tempAllX[:, index] = (tempAllX[:, index] - avgOfX)/(np.max(tempAllX[:, index] - np.min(tempAllX[:, index])))
 
    return tempAllX, np.array(outputData), inputData.head(0).columns.values#append to outputData in arg 2 if regress breaks , dtype="float"
    
    

def chooseFeatures(XFeatures, XFile,yFile):
    inputData = pd.read_csv(XFile,
                            sep=',',
                            usecols=XFeatures,
                            header=0)
    outputData = pd.read_csv(yFile,
                             sep=',',
                             usecols=[0],
                             header=0)

    tempAllX = np.array(inputData, dtype="float")
    
    #minmax scaling
    for index in range(tempAllX.shape[1]):
        avgOfX = np.average(tempAllX[:, index])

        tempAllX[:, index] = (tempAllX[:, index] - avgOfX)/(np.max(tempAllX[:, index] - np.min(tempAllX[:, index])))
 
    return tempAllX, np.array(outputData), inputData.head(0).columns.values