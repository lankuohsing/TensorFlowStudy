# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:04:13 2017

@author: lankuohsing
"""
import tensorflow as tf
# Create some variables.
weights = tf.Variable(tf.random_normal([5, 5], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([2]), name="biases")
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  # Save the variables to disk.
  save_path = saver.save(sess, "D:/Projects/Github/Python/TensorFlowTutorial/tmp/model.ckpt")
  print ("Model saved in file: ", save_path)