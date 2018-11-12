# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:23:12 2018

@author: Casey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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
    featureNames = []
    for index in range(tempAllX.shape[1]):
        avgOfX = np.average(tempAllX[:, index])

        tempAllX[:, index] = (tempAllX[:, index] - avgOfX)/(np.max(tempAllX[:, index] - np.min(tempAllX[:, index])))
 
    return tempAllX, np.array(outputData, dtype="float"), inputData.head(0).columns.values


def getPlotInfo(allX, allY,W,b):
    minSize = min(allX[:, 0])
    maxSize = max(allX[:, 0])
    
    minYear = min(allX[:, 1])
    maxYear = max(allX[:, 1])
    
    # np.ascalar(myModel.intercept_)
    
    
    point1X = np.array([minSize, minYear]).reshape(2, 1)
    point1Y = np.sum(point1X * W) + b
    
    point2X = np.array([minSize, maxYear]).reshape(2, 1)
    point2Y = np.sum(point2X * W) + b
    
    point3X = np.array([maxSize, minYear]).reshape(2, 1)
    point3Y = np.sum(point3X * W) + b
    
    point4X = np.array([maxSize, maxYear]).reshape(2, 1)
    point4Y = np.sum(point4X * W) + b
    
    allFourX1 = [point1X[0][0], point2X[0][0], point3X[0][0], point4X[0][0]]
    allFourX2 = [point1X[1][0], point2X[1][0], point3X[1][0], point4X[1][0]]
    
    allFourY = [point1Y, point2Y, point3Y, point4Y]


    
    return np.array(allFourX1),np.array(allFourX2),allFourY