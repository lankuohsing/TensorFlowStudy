# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:52:47 2017

@author: lankuohsing
"""

import tensorflow as tf
import os
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
tf.reset_default_graph()
v1 = tf.Variable([1,2], name="v1")
v2 = tf.Variable([3,4], name="v2")
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, BASE_DIR+"/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print(sess.run(v1))
    print(sess.run(v2))