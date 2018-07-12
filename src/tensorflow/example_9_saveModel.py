# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# Save to file
# remember to define the same dtype and shape where restore
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init = tf.initialize_all_variables()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,r"D:\machineLearning\src\tensorflow\my_net\save_net.ckpt")
#     print("Save to path:",save_path)

# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    # 加载的时候，回去加载同名的weights和biases
    saver.restore(sess,r"D:\machineLearning\src\tensorflow\my_net\save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))