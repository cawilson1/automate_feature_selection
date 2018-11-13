# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:12:03 2018

@author: Casey
"""
from process_data import chooseFeatures,readAllFeatures
from sklearn import linear_model,svm
from sklearn.model_selection import cross_val_score
from itertools import chain

def recursiveElim(startingFeatures,optimalSetSize,XFile,yFile,mlAlg):
    
    if len(startingFeatures) == optimalSetSize:#certain ml algs like svm may have better scores with less features at times than with all features. Implement this later
        return startingFeatures  #end recursion
    
    featureScores = []
    print('starting features for current round',startingFeatures)
    print('number of features left', len(startingFeatures))
    for index in range(len(startingFeatures)):
        
        tempFeatures = [startingFeatures[0:index]]
        tempFeatures.append(startingFeatures[index+1:])
        tempFeatures = list(chain.from_iterable(tempFeatures))#remove nested list and present as all one list
        print(tempFeatures)
        
        allX, allY, features = chooseFeatures(tempFeatures,XFile,yFile)                           
        non,non2,nonFeatures = chooseFeatures([startingFeatures[index]],XFile,yFile)#this is just a lazy way to get the feature name
       # scores = getModelScores('lin_reg',allX,allY,10)
       # scores = getModelScores('svm',allX,allY,5) 
        scores = getModelScores(mlAlg,allX,allY,10)
        print('error for this set of features',scores.mean())
        featureScores.append([scores.mean(),nonFeatures[0],startingFeatures[index]])
    sortedList = sorted(featureScores, reverse=False)#make boolean ifMinimizing
    print('\n\n\neliminate feature',(sortedList[-1][1]),'\n\n')
    sortedList = sortedList[:-1]#remove the worst performing feature
    startingFeatures = []#reset list for next recursion round
    for el in sortedList:
        startingFeatures.append(el[2])#append the feature indeces for later call to chooseFeatures
        
    startingFeatures = sorted(startingFeatures)#for consistency
    #print(startingFeatures)
    
    startingFeatures = recursiveElim(startingFeatures,optimalSetSize,XFile,yFile,mlAlg)
    return startingFeatures

def getModelScores(mlAlgorithm,X,y,folds):
    if mlAlgorithm == 'lin_reg':
        myModel = linear_model.LinearRegression()#here call a function that specifies ml model
        return cross_val_score(myModel,X,y,scoring='neg_mean_squared_error',cv=folds)
    elif mlAlgorithm == 'svm':
        y = y.reshape(len(y),)#reshape so svm doesn't complain
        myModel = svm.SVC(gamma='auto',kernel='rbf', decision_function_shape='ovo')#use decision_function_shape arg when using multiclass
        return cross_val_score(myModel, X, y, cv=5)#change cv later
    else:
        print('no match for ML algorithm')
        return 0
    
    
def specifyDataset(XFile,yFile,mlAlg,numFeatures):#if featuresList is empty, by default start with all features specified in dataset
    
    X,y,features = readAllFeatures(XFile,yFile)
    startingFeatures = range(len(X[0]))#create a list starting with all feature indeces in ascending order
    #next create 2 functions in process_data, the current one called pickFeatures, and the new one called readAllColumns. Maybe later give arg option to choose starting set
    #startingFeatures = [0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]#all relevant features. 1 is date and 2 is price (the dependent var)
    #allX, allY, features = chooseFeatures(startingFeatures, [2])                           
   
    scores = getModelScores(mlAlg,X,y,10)
    # myModel = linear_model.LinearRegression()
    #scores = cross_val_score(myModel,X,y,scoring='neg_mean_squared_error',cv=10,)
    #myModel = svm.SVC(gamma='auto',kernel='rbf', decision_function_shape='ovo')
   # y = y.reshape(len(y),)#reshape so svm doesn't complain
    #scores = cross_val_score(myModel,X,y,cv=10,)
    
    print('error for all features',scores.mean())#baseline score for using all features
    
    optimalFeatures = recursiveElim(startingFeatures,6,XFile,yFile,mlAlg)#second arg is what num of features to stop at
    print(optimalFeatures)
    
    allX, allY, features = chooseFeatures(optimalFeatures, XFile,yFile) 
    scores = getModelScores(mlAlg,allX,allY,10)
    #myModel = linear_model.LinearRegression()
    #scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error',cv=10,)
    print('scores for optimal features',scores.mean())
    
    print(list(features))
    
