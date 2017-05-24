# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:09:18 2017
变量：创建、初始化、保存和加载
@author: lankuohsing
"""
import tensorflow as tf
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()
# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  
  # Save the variables to disk.
  save_path = saver.save(sess, "D:/Projects/Github/Python/TensorFlowTutorial/tmp/model.ckpt")
  #save_path = saver.save(sess, dir_path+"/tmp1/model.ckpt")
  print ("Model saved in file: ", save_path)