# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:22:01 2017

@author: languoxing
"""
# In[]
import tensorflow as tf
# In[]
tf.reset_default_graph()
# Create some variables.
v1 = tf.Variable(1, name="v1")
v2 = tf.Variable(2, name="v2")
# In[]
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    print(sess.run(v1))
    print(sess.run(v2))
    # Save the variables to disk.
    save_path = saver.save(sess, "./tmp1/model.ckpt")
    print("Model saved in file: %s" % save_path)