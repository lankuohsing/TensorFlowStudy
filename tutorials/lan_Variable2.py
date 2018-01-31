# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:57:14 2017

@author: lankuohsing
"""
import tensorflow as tf
import os
# Create some variables.
v1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="v1")
v2 = tf.Variable(tf.zeros([200]), name="v2")
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, BASE_DIR+"/tmp1/model.ckpt")
  print ("Model restored.")
  # Do some work with the model