# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:45:30 2021

@author: lankuohsing
"""

# In[]
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# In[]
model=keras.Sequential(
        [
                layers.Dense(units=3,input_shape=(2,)),
                layers.Dense(units=4),
                ]

        )
print("Number of weights: ",len(model.weights))
print("Number of layers: ",len(model.layers))