# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:08:35 2017
探究name_scope的用法
@author: lankuohsing
"""

import tensorflow as tf
with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
  print (a.name)
  print (W.name)
  print (b.name)