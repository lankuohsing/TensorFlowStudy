# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 00:35:25 2017

@author: languoxing
"""
# In[]
import tensorflow as tf
# In[]
with variable_scope.variable_scope("tet1"):
    var3 = tf.get_variable("var3",shape=[2],dtype=tf.float32)
    print var3.name
    with variable_scope.variable_scope("tet2"):
        var4 = tf.get_variable("var4",shape=[2],dtype=tf.float32)
        print var4.name