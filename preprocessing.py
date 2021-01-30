#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:10:46 2020

@author: ege
"""

import pickle
import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import itertools as it


def load_subject_data_from_file(id):
    with open("S{0}/S{0}.pkl".format(id), 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def restructure_data(data):

    new_data = {'label': np.array(data['label']), "signal": {}}

    for device in data['signal']:
        for type in data['signal'][device]:
            for i in range(len(data['signal'][device][type][0])):
                signal_name = '_'.join([device, type, str(i)])
                signal = np.array([x[i] for x in data['signal'][device][type]])
                new_data["signal"][signal_name] = signal
                       
    return new_data
               

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=5, fs=64, order=3, start_from=1000):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return pd.Series(y[start_from:])

def original_sampling(channel_name: str):
    if channel_name.startswith("chest"):
        return 700
    if channel_name.startswith("wrist_BVP"):
        return 64
    if channel_name.startswith("wrist_ACC"):
        return 32
    if channel_name.startswith("wrist_EDA"):
        return 4
    if channel_name.startswith("wrist_TEMP"):
        return 4
   
def target_sampling(channel_name: str):
    if channel_name.startswith("chest_ECG"):
        return 70
    if channel_name.startswith("chest_ACC"):
        return 10
    if channel_name.startswith("chest_EMG"):
        return 10
    if channel_name.startswith("chest_EDA"):
        return 3.5
    if channel_name.startswith("chest_Temp"):
        return 3.5
    if channel_name.startswith("chest_Resp"):
        return 3.5
    if channel_name.startswith("wrist_ACC"):
        return 8
    if channel_name.startswith("wrist"):
        return original_sampling(channel_name)
    if channel_name == "label":
        return 700
   
def filter_signal(data):
    signals = data['signal']
    for signal_name in signals:  
        result = scipy.stats.mstats.winsorize(signals[signal_name], limits = [0.03,0.03])
        if original_sampling(signal_name)/2 > 10:
            result = butter_lowpass_filter(result, cutoff = 10, fs = original_sampling(signal_name), start_from = 0)
       
        result = pd.Series(result).iloc[::int(original_sampling(signal_name) / target_sampling(signal_name))]        
        result = np.array(result).reshape(-1,1)
       
        scaler = MinMaxScaler()
        result = scaler.fit_transform(result)
       
        signals[signal_name] = result
   

def indexes_for_signal(i,signal):
    if(signal == "label"):
        freq = 700
    else:
        freq = target_sampling(signal)
       
    first_index = int((i * freq) // 4)
    window_size = int(60 * freq)
    return first_index, first_index + window_size

def create_sliding_windows(data):
    X = [[] for signal in data["signal"]]
    Y = []
   
    for i in range (0, len(data["signal"]["wrist_EDA_0"]) - 240, 120):
        first_index, last_index = indexes_for_signal(i, "label")
        label_id = scipy.stats.mstats.mode(data["label"][first_index:last_index])[0][0]
       
        if label_id not in [1, 2, 3]:
            continue
       
        channel_id = 0
        for signal in data["signal"]:
            first_index, last_index = indexes_for_signal(i, signal)
            X[channel_id].append(data["signal"][signal][first_index:last_index])
            channel_id += 1
           
        Y.append(label_id)
       
    return X,Y


if __name__ == '__main__':
   
    SUBJECTS_IDS = list(it.chain(range(2, 12), range(13, 18)))
    for id in SUBJECTS_IDS:
        d = load_subject_data_from_file(id)
        d = restructure_data(d)
        filter_signal(d)
        globals()["X"+str(id)], globals()["Y"+str(id)] = create_sliding_windows(d)
        with open('preprocessed_X/X{0}'.format(id), 'wb') as fp:
            pickle.dump(globals()["X"+str(id)], fp)
        with open('preprocessed_Y/Y{0}'.format(id), 'wb') as fp:
            pickle.dump(globals()["Y"+str(id)], fp)