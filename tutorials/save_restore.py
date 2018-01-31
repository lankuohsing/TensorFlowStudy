# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 01:20:10 2017

@author: lankuohsing
"""

import tensorflow as tf
import os
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
with tf.Session() as sess:
    var = tf.Variable(42, name='var')
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(BASE_DIR+'file.meta')
tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('file.meta')
    var = tf.global_variables()[0]
    sess.run(tf.initialize_all_variables())
    print(sess.run(var))