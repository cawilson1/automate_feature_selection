# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:35:40 2019

@author: Casey
"""

#from recursive_feature_elimination import specifyDataset
import numpy as np
from process_data import readAllFeatures, chooseFeatures
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score

def oneVAll(attack, table):
    print('here withh ', attack)
    tempTable = np.array(table)
  #  print('r2l'==attack)
    for idx,el in enumerate(tempTable):
        if el != attack:
         #   print(el)
            tempTable[idx] = 0
        else:
            tempTable[idx]=1
           # if attack == 'r2l':
              #  print(idx)
            
    return tempTable

def rfe(X,y):
    pass

def parallelRFE(i,featureVals,labels):
    #reshape into a vector
    labels = labels.reshape(len(labels),)
    
    print('shape of featureVals', featureVals.shape)
    
    accuracyScores = []
    
    kf = KFold(len(labels),5)
   # print(kf)
    print('current feature: ',str(i))
    
    for train_index, test_index in kf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = featureVals[train_index], featureVals[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        myModel = svm.SVC(gamma=1.0,kernel='rbf')#,C=20.0)#for gamma: (1/n_feats) * stdX. In this case, n_feats is 1, so first part is ignored
       # myModel = svm.SVC(gamma=2.0,kernel='rbf',C=5.0)
        # print(y_train.shape)
        myModel.fit(X_train,y_train)
        #predict
        predicted = myModel.predict(X_test)
        non,non2,nonFeatures = chooseFeatures([index],'./datasets/kc_house_data/kc_house_data_X.csv','./datasets/kc_house_data/kc_house_data_X.csv')#this is just a lazy way to get the feature name
        
        print('accuracy minus feature',str(i), nonFeatures[0],accuracy_score(y_test,predicted))
        
        accuracyScores.append(accuracy_score(y_test,predicted))
        
    print(accuracyScores)

X,y,features = readAllFeatures('./datasets/kc_house_data/kc_house_data_X.csv','./datasets/kc_house_data/kc_house_data_y_classification_2_class.csv')

print(y)

y = oneVAll('average', y).reshape(len(y),)

print(y)
rfe(X,y)


for index in range(len(X[0])):
        
    tempFeatures = np.delete(X,[index],1)#all features in current round minus one (i.e. remove a column)
    print(tempFeatures)
    
    
#https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
    parallelRFE(index,tempFeatures,y.astype('int'))



#define the features to run before each parallel RFE call


#specifyDataset(X,y,'svm',15)
