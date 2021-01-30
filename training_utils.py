#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:12:39 2020

@author: ege
"""

import itertools as it
import random
import math
import pickle
import numpy as np


           
def n_fold_split():
    SUBJECTS_IDS = list(it.chain(range(2, 12), range(13, 18)))    
    sets = []
   
    random.seed(5)
    subject_ids = list(SUBJECTS_IDS)
    random.shuffle(subject_ids)
   
    test_sets = [subject_ids[i::5] for i in range(5)]
   
    for test_set in test_sets:
        remaining = [x for x in subject_ids if x not in test_set]
        val_set = random.sample(remaining,2)
        train_set = [x for x in remaining if x not in val_set]
        sets.append({"train": train_set, "test": test_set, "val": val_set})
   
    random.seed()
   
    return sets


def create_train_test_sets(sets,iteration):
    datasets_dict = {}
    x_train = [[] for i in range(14)]
    x_test = [[] for i in range(14)]
    x_val = [[] for i in range(14)]
    y_train = []
    y_test = []
    y_val = []
    for subject in sets[iteration]['train']:
        with open("preprocessed_X/X{0}".format(subject), 'rb') as f:
            d = pickle.load(f)
        for channel in range(len(d)):
            x_train[channel] += d[channel]
           
        with open("preprocessed_Y/Y{0}".format(subject), 'rb') as f:
            d_y = pickle.load(f)
            y_train += d_y
           
    for subject in sets[iteration]['test']:
        with open("preprocessed_X/X{0}".format(subject), 'rb') as f:
            d = pickle.load(f)
        for channel in range(len(d)):
            x_test[channel] += d[channel]
           
        with open("preprocessed_Y/Y{0}".format(subject), 'rb') as f:
            d_y = pickle.load(f)
            y_test += d_y
            
    for subject in sets[iteration]['val']:
        with open("preprocessed_X/X{0}".format(subject), 'rb') as f:
            d = pickle.load(f)
        for channel in range(len(d)):
            x_val[channel] += d[channel]
           
        with open("preprocessed_Y/Y{0}".format(subject), 'rb') as f:
            d_y = pickle.load(f)
            y_val += d_y
    
    x_train = [np.array(x) for x in x_train]
    x_test = [np.array(x) for x in x_test]
    x_val = [np.array(x) for x in x_val]
    
    datasets_dict["wesad"] = (x_train, y_train, x_test,
                                       y_test, x_val, y_val)
    return datasets_dict