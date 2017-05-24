# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:01:10 2017

@author: lankuohsing
"""

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)
  print(sess.run([mul, intermed]))

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]