# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:59:32 2018

@author: Casey
"""

from process_data import chooseFeatures,readAllFeatures
from run_ml_alg import getModelScores
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

    
def classifierForTopNFeatures(n,sortedFeatures,XFile,yFile,mlAlg):
    topFeatureIndeces = []
    for i in range(n):
        topFeatureIndeces.append(sortedFeatures[i][2])
    print(topFeatureIndeces)
    allX, allY, features = chooseFeatures(topFeatureIndeces, XFile,yFile)
    scores = getModelScores(mlAlg,allX,allY,10)
    print('error for top 6 features',features,scores.mean())
    
    
    #when selecting top 6 features, they are 5,11,12,19,4,9. This is run above
    #code below replaces 12 from above with 10 (a worse performing individual feature) for an overall better score combined
    
    #allX, allY, features = chooseFeatures([3,9,8,17,2,7], XFile,yFile)   
    #scores = getModelScores(mlAlg,allX,allY,10)
    

def specifyDataset(XFile,yFile,mlAlg,numFeatures):#if featuresList is empty, by default start with all features specified in dataset

    loopLength,non,non1 = readAllFeatures(XFile,yFile)# just to get length of x
    emptyList = []
    for i in range(len(loopLength[0])):
        if i != 1 and i != 2:
            allX,allY,features = chooseFeatures([i], XFile,yFile)
            scores = getModelScores(mlAlg,allX,allY,10)
            print(scores.mean(),features[0],i)
            
            emptyList.append([scores.mean(),features[0],i])
    
    
    sortedList = sorted(emptyList, reverse=True)
    
    
    for i in range(len(sortedList)):
        print(sortedList[i])
        
    classifierForTopNFeatures(6,sortedList,XFile,yFile,mlAlg)#first arg is number of top ranked features to run
