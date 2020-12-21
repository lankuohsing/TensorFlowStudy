# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:20:35 2020

@author: lankuohsing
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# In[]
def get_model():
    # create a linear regression model
    inputs=keras.Input(shape=(1,),name="InputX")
    outputs=keras.layers.Dense(1,name="Output")(inputs)
    model=keras.Model(inputs,outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
def plot_figure(X_test, predictions,labels,figure_name):
    #对测试数据的真实值和预测值绘图比较
    plt.figure()
    plt.plot(X_test,predictions,label="prediction_value")
    plt.plot(X_test,predictions,label="real_value")
    plt.legend()
    plt.title(figure_name)
    plt.savefig("figures/"+figure_name+"_tf2.png")
    plt.show()
    return
def keras2pb(keras_model,model_path,pb_name):
    from tensorflow.python.keras.saving import saving_utils as _saving_utils
    from tensorflow.python.framework import convert_to_constants as _convert_to_constants
    from tensorflow.compat.v1 import graph_util
    func=_saving_utils.trace_model_call(keras_model)
    concrete_func=func.get_concrete_function()
    frozen_func, graph_def=(_convert_to_constants.convert_variables_to_constants_v2_as_graph(concrete_func,lower_control_flow=False))
    graph=graph_util.remove_training_nodes(graph_def)
    tf.io.write_graph(graph,model_path,pb_name,as_text=False)
    return
# In[]
model=get_model()
a=0
b=10
X_train=(b-a)*np.random.random((1000,1))+a
Y_train=2*X_train+3
X_test=(b-a)*np.random.random((1000,1))+a
Y_test=2*X_test+3
# Train the model
model.fit(X_train,Y_train,batch_size=8,epochs=200)
# In[]
keras2pb(model,"./models/","linear_regression.pb")
model.save("./models/linear_regression")
# In[]
predictions=model.predict(X_test)
plot_figure(X_test,predictions,Y_test,"y=2x+3")
# In[]
# rescontructed_model
reconstrcted_model=keras.models.load_model("./models/linear_regression")
np.testing.assert_allclose(model.predict(X_train),reconstrcted_model.predict(X_train))
"""
reconstrcted_model.fit(X_train,Y_train)
"""
# In[]