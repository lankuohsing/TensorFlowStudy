# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:04:15 2017

@author: lankuohsing
"""

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# 输出:
# [array([ 14.], dtype=float32)]