# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:59:27 2017

@author: lankuohsing
"""
import tensorflow as tf
import os
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
sess = tf.Session()
new_saver = tf.train.import_meta_graph(BASE_DIR+"/tmp2/model.meta")
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)