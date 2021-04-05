# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:30:08 2021

@author: lankuohsing
"""

# In[]
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# In[]
model=keras.Sequential(
        [
                layers.InputLayer(input_shape=(2,),name="InputX"),
                layers.Dense(units=3),
                layers.Dense(units=4),
                ]

        )
print("Number of weights: ",len(model.weights))
print("Number of layers: ",len(model.layers))