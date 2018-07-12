# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

coefficients = np.array([[1.],[-20.],[100.]])

w = tf.Variable(0,dtype=tf.float32) # 将w初始化为0
x = tf.placeholder(tf.float32,[3,1])
# cost function w^2-10w+25
# cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
# cost = w**2 - 10*w + 25
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
# 0.01 是学习率，minimize是最小化损失函数
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init) # 初始化全局变量
# print(session.run(w))

session.run(train,feed_dict={x:coefficients}) # 进行一次梯度下降
print(session.run(w))

# 执行1000次梯度下降
for i in range(1000):
    session.run(train,feed_dict={x:coefficients})

print(session.run(w))