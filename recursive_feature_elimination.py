# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:12:03 2018

@author: Casey
"""
from process_data import readFile
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from itertools import chain

def recursiveElim(startingFeatures,optimalSetSize):
    
    if len(startingFeatures) == optimalSetSize:
        return startingFeatures  #end recursion
    
    featureScores = []
    print('starting features for current round',startingFeatures)
    print('number of features left', len(startingFeatures))
    for index in range(len(startingFeatures)):
        
        tempFeatures = [startingFeatures[0:index]]
        tempFeatures.append(startingFeatures[index+1:])
        tempFeatures = list(chain.from_iterable(tempFeatures))#remove nested list and present as all one list
        print(tempFeatures)
        
        allX, allY, features = readFile(tempFeatures, [2])                           
        non,non2,nonFeatures = readFile([startingFeatures[index]],[2])
        scores = getModelScores('lin_reg',allX,allY,10)
        print('error for this set of features',scores.mean())
        featureScores.append([scores.mean(),nonFeatures[0],startingFeatures[index]])
    sortedList = sorted(featureScores, reverse=False)#make boolean ifMaximizing
    print('\n\n\neliminate feature',(sortedList[-1][1]),'\n\n')
    sortedList = sortedList[:-1]#remove the worst performing feature
    startingFeatures = []#reset list for next recursion round
    for el in sortedList:
        startingFeatures.append(el[2])#append the feature indeces for later call to readFile
        
    startingFeatures = sorted(startingFeatures)#for consistency
    #print(startingFeatures)
    
    startingFeatures = recursiveElim(startingFeatures,optimalSetSize)
    return startingFeatures

def getModelScores(mlAlgorithm,X,y,folds):
    if mlAlgorithm == 'lin_reg':
        myModel = linear_model.LinearRegression()#here call a function that specifies ml model
        return cross_val_score(myModel,X,y,scoring='neg_mean_squared_error',cv=folds)
        
    else:
        print('no match for ML algorithm')
        return 0
    
    
def specifyDataset(dataset,X,y,numFeatures):#if featuresList is empty, by default start with all features specified in dataset
    
    #next create 2 functions in process_data, the current one called pickFeatures, and the new one called readAllColumns. Maybe later give arg option to choose starting set
    startingFeatures = [0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]#all relevant features. 1 is date and 2 is price (the dependent var)
    allX, allY, features = readFile(startingFeatures, [2])                           
    myModel = linear_model.LinearRegression()
    myModel.fit(allX, allY)
    scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error',cv=10,)
    print('error for all features',scores.mean())#baseline score for using all features
    
    optimalFeatures = recursiveElim(startingFeatures,6)#second arg is what num of features to stop at
    print(optimalFeatures)
    
    allX, allY, features = readFile(optimalFeatures, [2])                             
    myModel = linear_model.LinearRegression()
    myModel.fit(allX, allY)
    scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error',cv=10,)
    print('error for optimal features',scores.mean())
    
    print(list(features))
    
