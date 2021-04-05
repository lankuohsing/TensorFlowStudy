# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:44:09 2021

@author: lankuohsing
"""
'''recommend'''
# In[]
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# In[]
model=keras.Sequential(
        [
                layers.Input(shape=(2,),name="InputX"),
                layers.Dense(units=3),
                layers.Dense(units=4),
                ]

        )
print("Number of weights: ",len(model.weights))
print("Number of layers: ",len(model.layers))