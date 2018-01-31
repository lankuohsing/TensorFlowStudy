# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 01:23:39 2017

@author: lankuohsing
"""

import tensorflow as tf
import os
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(BASE_DIR+'/tmp3/file.meta')
    var = tf.global_variables()[0]
    sess.run(tf.global_variables_initializer())
    print(sess.run(var))