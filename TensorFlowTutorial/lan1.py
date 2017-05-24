# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:14:38 2017

@author: lankuohsing
"""


import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))
print(5/2)
