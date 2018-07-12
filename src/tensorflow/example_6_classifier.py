# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True) # 若电脑上没有这个数据包，则会从网络上下载下来


def add_layer(inputs,in_size,out_size,activation_funcation=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plub_b = tf.matmul(inputs,Weights) + biases

    if activation_funcation is None:
        outputs = Wx_plub_b
    else:
        outputs = activation_funcation(Wx_plub_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    '''
        tf.arg_max：
        返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，
        那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
        0表示的是按列比较返回最大值的索引，1表示按行比较返回最大值的索引
    '''
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])  # 每张图片有784个像素点（28*28）
ys = tf.placeholder(tf.float32,[None,10]) # 每个样本都有10个输出

# add output layer
prediction = add_layer(xs,784,10,activation_funcation=tf.nn.softmax)

# the error between prediction and real data
# 这里使用交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1])) # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100) # 每次学习100个训练样本
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))


