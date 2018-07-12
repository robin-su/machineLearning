# -*- coding:utf-8 -*-

import tensorflow as tf

# 使用placeholder意味着你要做的事情是在运行的时候给他赋值

#input1 = tf.placeholder(tf.float32,[2,2]) # 要给定type,大部分情况只能处理float32的数据,规定他是两行两列的结构
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]})) # 将placeholder类型的变量input1和input2赋值成7.0,2.0
