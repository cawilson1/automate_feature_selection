# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:11:34 2018

@author: Casey
file to test command line stuff
"""
import sys
import manually_choose_features
import single_feature_classifier
import recursive_feature_elimination

def main():
    input = sys.argv
    #input[0] is file name
    
    featureIndexList = []
    #later, specify model (i.e. regression) and data set.
    #if model is regression, specify feature index to compare
    #if model is classification, specify classes somehow, either manually, through a file, create dynamically, etc.
    #give option for cross val and num of folds
    #for relevant options, specify number of features to stop at
    #add some normalization options and specify a default
    #ideally should have input validation
    if(input[1] == '-manual'):#the person wants to manually enter feature indeces. Probably not commonly recommended
        print('manual entry')
        for el in input[2:]:
            featureIndexList.append(int(el))
        manually_choose_features.enterFeatureIndeces(featureIndexList)
    elif(input[1] == '-sfc'):#single feature classifier
        print('single feature classifier')
        single_feature_classifier.specifyDataset('datasetname.csv',[])
    elif(input[1] == '-rfe'):
        print('recursive feature elimination')
        recursive_feature_elimination.specifyDataset('datasetname.csv',[])


if __name__=="__main__":
    main()
    exit()