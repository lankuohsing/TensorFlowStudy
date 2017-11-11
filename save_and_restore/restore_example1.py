# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:28:02 2017

@author: languoxing
"""
# In[]
import tensorflow as tf
tf.reset_default_graph()#注意，一定要有这个，不然会报错
# In[]
# Create some variables.
v1 = tf.Variable(3, name="v1")
v2 = tf.Variable(4, name="v2")
# In[]
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()
#init_op = tf.global_variables_initializer()
# In[]
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    print(sess.run(v1))
    print(sess.run(v2))
    # Restore variables from disk.
    saver.restore(sess, "./tmp1/model.ckpt")
    print("Model restored from file: %s" % save_path)
    # Do some work with the model
    print(sess.run(v1))
    print(sess.run(v2))