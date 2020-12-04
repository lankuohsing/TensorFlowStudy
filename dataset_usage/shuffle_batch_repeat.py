# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 21:08:13 2020

@author: lankuohsing
"""
import tensorflow as tf
import numpy as np
# https://www.cnblogs.com/wisir/p/13497261.html
# In[]

ori_data = np.arange(20).reshape((4, 5))
ds = tf.data.Dataset.from_tensor_slices(ori_data)
print(ori_data)

'''
shuffle: 维持一个buffer_size大小的缓存，打乱后供后续打包成batch输出
具体来说，从data数据集中按顺序抽取buffer_size个样本放在buffer中，然后打乱buffer中的样本
buffer中样本个数不足buffer_size，继续从data数据集中按顺序填充至buffer_size，
此时会再次打乱
batch: 打包成一个batch
repeat: 重复多次，构造成多个epoch
'''
# In[]
def f1(ds):
    # 最常用的顺序
    # 解释：相当于把所有数据先打乱，然后打包成batch输出，整体数据重复2个epoch
    # 特点：1.一个batch中的数据不会重复；2.每个epoch的最后一个batch的尺寸小于等于batch_size
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(3)
    ds = ds.repeat(count=2)
    # 构造获取数据的迭代器
    iters = ds.make_one_shot_iterator()
    # 每次从迭代器中获取一批数据
    batch = iters.get_next()
    sess = tf.Session()
    # 数据集完成遍历完之后，继续抽取的话会报错：OutOfRangeError
    for i in range(0,4):
        print(i)
        print(sess.run(batch))
# In[]
def f2(ds):
    # 解释：相当于把所有数据先打乱，再把所有数据重复两个epoch，然后将重复两个epoch的数据放在一起，最后打包成batch_size输出
    # 特点：1.因为把数据复制两份，还进行打乱，因此某个batch数据可能会重复，而且出现重复数据的batch只会是两个batch交叉的位置；2.最后一个batch的尺寸小于等于batch_size
    ds = ds.shuffle(buffer_size=100)
    ds = ds.repeat(count=2)
    ds = ds.batch(3)
    # 构造获取数据的迭代器
    iters = ds.make_one_shot_iterator()
    # 每次从迭代器中获取一批数据
    batch = iters.get_next()
    sess = tf.Session()
    # 数据集完成遍历完之后，继续抽取的话会报错：OutOfRangeError
    for i in range(0,3):
        print(i)
        print(sess.run(batch))
# In[]
def f3(ds):
    # 解释：相当于把所有数据先打包成batch，然后把打包成batch的数据重复两遍，最后再将所有batch打乱进行输出
    # 1.打乱的是batch；2.某些batch的尺寸小于等于batch_size，因为是对batch进行打乱，所以这些batch不一定是最后一个
    ds = ds.batch(3)
    ds = ds.repeat(count=2)
    ds = ds.shuffle(buffer_size=100)
    # 构造获取数据的迭代器
    iters = ds.make_one_shot_iterator()
    # 每次从迭代器中获取一批数据
    batch = iters.get_next()
    sess = tf.Session()
    # 数据集完成遍历完之后，继续抽取的话会报错：OutOfRangeError
    for i in range(0,4):
        print(i)
        print(sess.run(batch))
# In[]
def f4(ds):
    # 解释：相当于把所有数据先打包成batch，然后再将所有batch打乱打，最后包成batch的数据重复两遍并输出
    # 1.打乱的是batch；2.某些batch的尺寸小于等于batch_size，因为是对batch进行打乱，所以这些batch不一定是最后一个
    ds = ds.batch(3)
    ds = ds.shuffle(buffer_size=100)
    ds = ds.repeat(count=2)
    # 构造获取数据的迭代器
    iters = ds.make_one_shot_iterator()
    # 每次从迭代器中获取一批数据
    batch = iters.get_next()
    sess = tf.Session()
    # 数据集完成遍历完之后，继续抽取的话会报错：OutOfRangeError
    for i in range(0,4):
        print(i)
        print(sess.run(batch))
# In[]
def f5(ds):
    # 解释：相当于把所有数据先重复两遍，然后打乱，最后打包成batch
    # 1.某些batch的数据可能重复；2最后一个batch的尺寸小于等于batch_size.
    ds = ds.repeat(count=2)
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(3)

    # 构造获取数据的迭代器
    iters = ds.make_one_shot_iterator()
    # 每次从迭代器中获取一批数据
    batch = iters.get_next()
    sess = tf.Session()
    # 数据集完成遍历完之后，继续抽取的话会报错：OutOfRangeError
    for i in range(0,3):
        print(i)
        print(sess.run(batch))
# In[]
def f6(ds):
    # 解释：相当于把所有数据先重复两遍，然后打包成batch，最后打乱
    # 1.batch内部的数据不会重复；2.某一个batch的尺寸小于等于batch_size，但是打乱了所以不一定在最后一个.
    ds = ds.repeat(count=2)
    ds = ds.batch(3)
    ds = ds.shuffle(buffer_size=100)


    # 构造获取数据的迭代器
    iters = ds.make_one_shot_iterator()
    # 每次从迭代器中获取一批数据
    batch = iters.get_next()
    sess = tf.Session()
    # 数据集完成遍历完之后，继续抽取的话会报错：OutOfRangeError
    for i in range(0,3):
        print(i)
        print(sess.run(batch))
