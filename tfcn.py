#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:27:50 2020

@author: ege
"""

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.backend import cast
import copy


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
        res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
        res['precision'] = precision_score(y_true, y_pred, average='macro')
        res['accuracy'] = accuracy_score(y_true, y_pred)
        res['recall'] = recall_score(y_true,y_pred,average='macro')
        res['f1_score'] = f1_score(y_true,y_pred,average='macro')
        
        return res
    
        if not y_true_val is None:
            
            res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)
            res['recall'] = recall_score(y_true, y_pred, average='macro')
            res['duration'] = duration
            
            return res

class Classifier_TFCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory+'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        input_layers = []
        channel_outputs = []
        
        for shape in input_shape:
            current_layer = keras.layers.Input(shape)
            input_layers.append(current_layer)
            
            current_layer = self.build_FCN_block(current_layer)
            
            conv_x = keras.layers.Conv1D(filters = 6, kernel_size = 7, padding = 'valid', activation='relu')(current_layer)
            conv_x = keras.layers.AveragePooling1D(pool_size=3)(conv_x)
        
            conv_y = keras.layers.Conv1D(filters = 12, kernel_size = 7, padding = 'valid', activation='relu')(conv_x)
            conv_y = keras.layers.AveragePooling1D(pool_size=3)(conv_y)
            
            flatten_layer = keras.layers.Flatten()(conv_y)
            channel_outputs.append(flatten_layer)
            
            
        flat = keras.layers.concatenate(copy.copy(channel_outputs), axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(flat)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory+'best_model.hdf5'
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model
    
    def build_FCN_block(self,input_layer):
              
        conv1 = keras.layers.Conv1D(filters=64, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        
        shortcut_y = keras.layers.Conv1D(filters=64, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        
        output_block = keras.layers.add([shortcut_y, conv3])
        
        return(keras.layers.Activation('relu')(output_block))
        

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
		
        batch_size = 16
        nb_epochs = 1

        mini_batch_size = int(min(x_train[0].shape[0]/10, batch_size))

        start_time = time.time() 

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory+'best_model.hdf5')

        y_pred = self.model.predict(x_test)

        y_pred = np.argmax(y_pred , axis=1)

        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)
        
        keras.backend.clear_session()
        

