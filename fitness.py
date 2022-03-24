# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:26:08 2022

@author: ar2806
"""
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
import math
from sklearn.model_selection import train_test_split


from utils import load_dataset, prep_pixels
from activation import get_activation


def fitness(population):
    scores =[]
    trainX, trainY, valX, valY, testX, testY = load_dataset()
    #repare pixel data
    trainX, valX, testX = prep_pixels(trainX, valX, testX)
    
    for i,serial in zip(population,range(0,30)):
    #custom activation function
        
        def custom_activation(x):
            value = get_activation(i,x)
            return value
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Dense(32, activation=custom_activation))
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(32, activation=custom_activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(64, activation=custom_activation))
        model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(64, activation=custom_activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(128, activation=custom_activation))
        model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(128, activation=custom_activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer='he_uniform'))
        model.add(Dense(128, activation=custom_activation))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))

        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        


        print("Choromosome Number: ",serial, i)
        model.fit(trainX, trainY, epochs=1, batch_size=64, validation_data=(valX, valY), verbose=1)
        
        loss_check, acc_check = model.evaluate(testX, testY, verbose=0)
        if math.isnan(loss_check):
            scores.append(0.0)
            continue
        elif acc_check<0.25:
            scores.append(acc_check/loss_check)
            continue

        model.fit(trainX, trainY, epochs=14, batch_size=64, validation_data=(valX, valY), verbose=1)


        loss, acc = model.evaluate(testX, testY, verbose=0)
        
        scores.append(acc/loss)
        print("Test Score", acc/loss)
    return population, scores