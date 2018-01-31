# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:50:02 2017

@author: lankuohsing
"""
import tensorflow as tf
import os
# Create some variables.
v1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="v1")
v2 = tf.Variable(tf.zeros([200]), name="v2")
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  
  # Save the variables to disk.
  save_path = saver.save(sess, BASE_DIR+"/tmp1/model.ckpt")
  print("Model saved in file: ", save_path)


