# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 01:22:26 2017

@author: lankuohsing
"""

import tensorflow as tf
import os
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
with tf.Session() as sess:
    var = tf.Variable(42, name='var')
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(BASE_DIR+'/tmp3/file.meta')