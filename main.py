#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:12:44 2020

@author: ege
"""

from  training_utils import create_train_test_sets, n_fold_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tfcn import Classifier_TFCN

if __name__ == '__main__':
   
    sets = n_fold_split()
    
    for i in range(5):
        dataset_dict = create_train_test_sets(sets,i)
    
        x_train = dataset_dict['wesad'][0]
        y_train = dataset_dict['wesad'][1]
        x_test = dataset_dict['wesad'][2]
        y_test = dataset_dict['wesad'][3]
        x_val = dataset_dict['wesad'][4]
        y_val = dataset_dict['wesad'][5]
    
        nb_classes = len(np.unique(np.concatenate((y_train, y_test, y_val), axis=0)))
    
        enc = OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test, y_val), axis=0).reshape(-1, 1))
        y_train = np.array(y_train)
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = np.array(y_test)
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        y_val = np.array(y_val)
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

        y_true = np.argmax(y_test, axis=1)
        
    
        if type(x_train) == list:
            input_shapes = [x.shape[1:] for x in x_train]
        else:
            input_shapes = x_train.shape[1:]
        
    
        output_folder = "results{0}/".format(i)
    
        classifier = Classifier_TFCN(output_folder,input_shapes,nb_classes)
    
        classifier.fit(x_train,y_train, x_val, y_val, x_test, y_true)
