# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 10:53:23 2017

@author: languoxing
"""
# In[]
import tensorflow as tf
# In[]
with tf.variable_scope("scope1",reuse=True):
    w1 = tf.get_variable("w1", shape=[])
    w2 = tf.Variable(0.0, name="w2")
with tf.variable_scope("scope2", reuse=False):
    w1_p = tf.get_variable("w1", shape=[])
    w2_p = tf.Variable(1.0, name="w2")
# In[]
print(w1 is w1_p, w2 is w2_p)
print(w1.name)
print(w2.name)
print(w1_p.name)
print(w2_p.name)
#输出
#True  False