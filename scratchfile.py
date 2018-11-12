# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:29:17 2018

@author: Casey
file for testing code
"""


import pandas as pd
import numpy as np

def readFile(_inputColNums, _outputColNums, relativeFilename):
    inputData = pd.read_csv(relativeFilename,
                            sep=',',
                            usecols=_inputColNums,
                            header=0)
    outputData = pd.read_csv(relativeFilename,
                             sep=',',
                             usecols=_outputColNums,
                             header=0)

    tempAllX = np.array(inputData, dtype="float")
    
    #minmax scaling
    for index in range(tempAllX.shape[1]):
        avgOfX = np.average(tempAllX[:, index])

        tempAllX[:, index] = (tempAllX[:, index] - avgOfX)/(np.max(tempAllX[:, index] - np.min(tempAllX[:, index])))
 
    return tempAllX, np.array(outputData, dtype="float"), inputData.head(0).columns.values


def main():
    filename = "./datasets/kc_house_data/kc_house_data.csv"
    X, y, features = readFile([0,3,4,5],[2],filename)
    #0-100,000 cheap
    #100,001-250,000 average
    #250,001-500,000 pricey
    #500,001-1,000,000 expensive
    #1,000,001 - 5,000,000 very expensive
    #5,000,000+ cray
    
    classList = []
    
    for i in range(len(y)):
        if y[i] <0:
            print('breaking')
            break
        elif y[i] < 100000:
            classList.append('cheap')
        elif y[i] < 250000:
            classList.append('average')
        elif y[i] < 500000:
            classList.append('pricey')
        elif y[i] < 1000000:
            classList.append('expensive')
        elif y[i] < 5000000:
            classList.append('v_expensive')
        else:
            classList.append('cray')
        
    print(classList[0])
    dataDict = {'price_class': classList}
    
    df = pd.DataFrame(data=dataDict)
    df.to_csv('./datasets/kc_house_data/kc_house_data_y_classification.csv',encoding='utf-8',index=False)
    
    print(features)
    
main()