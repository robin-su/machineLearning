# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ####

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) # 1表示1维，范围是-1.0到1.0
bias = tf.Variable(tf.zeros([1]))

y = Weights * x_data + bias

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5表示学习率
train = optimizer.minimize(loss)
# 在神经网络中初始化变量
init = tf.initialize_all_variables()
### create tensorflow structure end ####

sess = tf.Session()
sess.run(init) # Very importantssss

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(bias))