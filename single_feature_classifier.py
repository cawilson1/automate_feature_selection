# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:59:32 2018

@author: Casey
"""

from process_data import chooseFeatures
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

    
def classifierForTopNFeatures(n,sortedFeatures):
    topFeatureIndeces = []
    for i in range(n):
        topFeatureIndeces.append(sortedFeatures[i][2])
    print(topFeatureIndeces)
    allX, allY, features = chooseFeatures(topFeatureIndeces, [2])
    myModel = linear_model.LinearRegression()
    myModel.fit(allX, allY)
    scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error',cv=10)
    print('error for top 6 features',features,scores.mean())
    
    
    #when selecting top 6 features, they are 5,11,12,19,4,9. This is run above
    #code below replaces 12 from above with 10 (a worse performing individual feature) for an overall better score combined
    
    allX, allY, features = chooseFeatures([5,11,10,19,4,9], [2])                              
    myModel = linear_model.LinearRegression()
    myModel.fit(allX, allY)
    scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error',cv=10,)
   # print('error for not top 6 features',features,scores.mean())
    

def specifyDataset(dataset,featuresList):#if featuresList is empty, by default start with all features specified in dataset

    emptyList = []
    for i in range(21):
        if i != 1 and i != 2:
            allX,allY,features = chooseFeatures([i],[2])
            myModel = linear_model.LinearRegression()
            myModel.fit(allX, allY)
            scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error',cv=10,)
            print(scores.mean(),features[0],i)
            
            emptyList.append([scores.mean(),features[0],i])
    
    
    sortedList = sorted(emptyList, reverse=True)
    
    
    for i in range(len(sortedList)):
        print(sortedList[i])
        
    classifierForTopNFeatures(6,sortedList)#first arg is number of top ranked features to run
