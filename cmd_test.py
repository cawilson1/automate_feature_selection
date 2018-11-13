# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:11:34 2018

@author: Casey
file to test command line stuff
"""
import sys
import ast
import manually_choose_features
import single_feature_classifier
import recursive_feature_elimination

def main():
    input = sys.argv
    #input[0] is file name
    
    #later, specify model (i.e. regression) and data set.
    #if model is regression, specify feature index to compare
    #if model is classification, specify classes somehow, either manually, through a file, create dynamically, etc.
    #give option for cross val and num of folds
    #for relevant options, specify number of features to stop at
    #add some normalization options and specify a default
    #add a decision tree
    #ideally should have input validation
    #adding exceptions for args below would be nice
    #create an option to try all feature selection methods with given ml algs for best feature set overall
    #consider having separate data processing and feature selection components
    #it may be interesting to allow for variables entered from the command line to be used as x an y as well
    if(input[1] == '-manual'):#the person wants to manually enter feature indeces. Probably not commonly recommended
        
        mlAlg = input[2]
        
        XFile = './datasets/' + input[3]
        XFeatures = ast.literal_eval(input[4])
        yFile = './datasets/' + input[5]
        yFeature = ast.literal_eval(input[6])
        

        manually_choose_features.enterFeatureIndeces(XFeatures,yFeature,XFile,yFile,mlAlg)
    elif(input[1] == '-sfc'):#single feature classifier
        mlAlg = input[2]
        checkMLAlg(mlAlg)
        XFile = './datasets/' + input[3]
        yFile = './datasets/' + input[4]
        finalFeatureSetSize = input[5]#check that this is an int
        print('single feature classifier\n')
        single_feature_classifier.specifyDataset(XFile,yFile,mlAlg,finalFeatureSetSize)
    elif(input[1] == '-rfe'):
        
        mlAlg = input[2]
        checkMLAlg(mlAlg)
        XFile = './datasets/' + input[3]
        yFile = './datasets/' + input[4]
        finalFeatureSetSize = input[5]#check that this is an int
        print('recursive feature elimination\n')
        recursive_feature_elimination.specifyDataset(XFile,yFile,mlAlg,finalFeatureSetSize)

def checkMLAlg(_mlAlg):
    if _mlAlg == 'lin_reg':
        return
    elif _mlAlg == 'svm':
        return
    
    else:
        printMLAlgOptions()
        exit()
    
def printMLAlgOptions():
    print("Machine learning algorithms to choose from and their argument names:\n")
    print('linear regression:  \"lin_reg\"')
    print('support vector machine: \"svm\"')

if __name__=="__main__":
    main()
    exit()