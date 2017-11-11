# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:35:24 2017

@author: languoxing
"""
# In[]
import tensorflow as tf
# In[]

checkpoint_dir = "./tmp2/model2"
# In[]
# first creat a simple graph
graph = tf.Graph()

#define a simple graph
with graph.as_default():
    x = tf.placeholder(tf.float32,shape=[],name='input')
    y = tf.Variable(initial_value=0,dtype=tf.float32,name="y_variable")
    update_y = y.assign(x)
    saver = tf.train.Saver(max_to_keep=3)
    init_op = tf.global_variables_initializer()
# In[]
# train the model and save the model every 400 iterations.
with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    for i in range(1,10000):
        y_result = sess.run(update_y,feed_dict={x:i})
        if i %4000 == 0:
            saver.save(sess,checkpoint_dir,global_step=i)    