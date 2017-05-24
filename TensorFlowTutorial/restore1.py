# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:54:27 2017

@author: lankuohsing
"""

import tensorflow as tf
import os
BASE_DIR = os.getcwd() #获取当前文件夹的绝对路径
tf.reset_default_graph()
v1 = tf.Variable([0,0], name="v1")
v2 = tf.Variable([0,0], name="v2")

saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, BASE_DIR+"/tmp/model.ckpt")
    print("Model restored.")
    # Do some work with the model
    print(sess.run(v1))
    print(sess.run(v2))