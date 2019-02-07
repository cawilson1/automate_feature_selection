# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:27:59 2018

@author: Casey
"""

from joblib import Parallel, delayed
import multiprocessing

inputs = range(10) 
def processInput(i,z):
    print(z)
    return z * i

num_cores = multiprocessing.cpu_count()
print(num_cores)
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,5) for i in inputs)
#results = Parallel(n_jobs=num_cores)(for i in inputs: delayed(processInput(i)))
print(results)


    num_cores = multiprocessing.cpu_count()
    var = Parallel(n_jobs=num_cores)(delayed(singVarClassifier)(i,XFile,yFile,mlAlg,numFeatures) for i in range(len(loopLength[0])))
    emptyList.append(var)