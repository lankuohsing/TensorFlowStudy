# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:40:25 2017

@author: lankuohsing

建立mnist前馈神经网络（Feedforward neural network）.
Implements the inference/loss/training pattern for model building.
1. inference()，推理 - Builds the model as far as is required for running the
network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
This file is used by the various "fully_connected_*.py" files and not meant to
be run，被"fully_connected_feed*.py"调用，不用单独运行.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
# MNIST数据集有10类
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
# 28×28像素
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.
  Args:(输入参数)
    images: Images placeholder, from inputs().图像占位符
    hidden1_units: Size of the first hidden layer.第一层隐藏层
    hidden2_units: Size of the second hidden layer.第二层隐藏层
  Returns:
    softmax_linear: Output tensor with the computed logits.输出计算逻辑张量
  """

    # Hidden 1,第一层隐藏层实现
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2,第二层隐藏层实现
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits  # 返回输出值


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.计算logits与标签的损失
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))