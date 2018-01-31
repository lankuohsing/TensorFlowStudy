# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:58:58 2017

@author: lankuohsing
"""
import tensorflow as tf
import os
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
w2 = tf.Variable(tf.truncated_normal(shape=[20]), name='w2')
tf.add_to_collection('vars', w1)
tf.add_to_collection('vars', w2)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, BASE_DIR+"/tmp2/model")
# `save` method will call `export_meta_graph` implicitly.
# you will get saved graph files:my-model.meta