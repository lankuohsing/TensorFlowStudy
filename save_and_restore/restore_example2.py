# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:25:42 2017

@author: languoxing
"""
# In[]
import tensorflow as tf
# In[]
tf.reset_default_graph()
restore_graph = tf.Graph()
with tf.Session(graph=restore_graph) as restore_sess:
    restore_saver = tf.train.import_meta_graph('./tmp2/model2-8000.meta')
    restore_saver.restore(restore_sess,tf.train.latest_checkpoint('./tmp2/'))
    print(restore_sess.run("y_variable:0"))
