# -*- coding:utf-8 -*-

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)
y_hat = tf.constant(36,name='y_hat') # 使用tensorflow定义一个36常量
y = tf.constant(39,name='y')

loss = tf.Variable((y-y_hat)**2,name='loss')

#添加节点用于初始化所有的变量。在你构建完整个模型并在会话中加载模型后，运行这个节点。
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)

sess= tf.Session()
print(sess.run(c))

x = tf.placeholder(tf.int64,name='x')
print(sess.run(2*x,feed_dict={x:3}))
sess.close()

def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()
    return result

print("result=" + str(linear_function()))

def sigmoid(z):

    x = tf.placeholder(tf.float32,name="x")
    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})

    return result

print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(12) = " + str(sigmoid(12)))

def cost(logits,labels):

    z = tf.placeholder(tf.float32,name="z")
    y = tf.placeholder(tf.float32,name="y")
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=z,labels=y)
    sess = tf.Session()
    cost = sess.run(cost,feed_dict={z:logits,y:labels})
    sess.close()
    return cost

logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits,np.array([0,0,1,1]))
print("cost = " + str(cost))

def one_hot_matrix(labels,C):

    C = tf.constant(C,name="C")

    one_hot_matrix = tf.one_hot(labels,C,axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels,C=4)
print("one_hot = " + str(one_hot))

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)

    sess.close()
    return ones
print ("ones = " + str(ones([3])))


