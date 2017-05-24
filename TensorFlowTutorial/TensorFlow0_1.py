# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:08:34 2017

@author: lankuohsing
"""

# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
y = tf.Variable([[10,2,3],[3,4,5],[5,4,3]])
# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()
y.initializer.run()
# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.sub(x, a)
miny=tf.argmax(y,1)#1代表行
print(sub.eval())
print(y.eval())
print(miny.eval())
# ==> [-2. -1.]
