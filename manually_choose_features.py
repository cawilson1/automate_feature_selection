# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:20:11 2018

@author: Casey

This file just runs whichever features you specify. "Domain expertise"
"""

from process_data import readFile
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

def enterFeatureIndeces(indexList):
    
    allX,allY,features = readFile(indexList, [2])
    
    #scores from example
    #allX,allY,features = readFile([3,4,5,6,10,16], [2])#enter indeces of features to run in first list
    
    myModel = linear_model.LinearRegression()
    myModel.fit(allX,allY)
    scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error', cv=10)#10 runs, each run has a different 90% of data for training, 10% for testing.
    print(scores.mean())#take the mean score of all cross val runs