# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=False)
'''
    epochs被定义为向前和向后传播中所有批次的单次训练迭代。
    这意味着1个周期是整个输入数据的单次向前和向后传递。
    简单说，epochs指的就是训练过程中数据将被“轮”多少次，就这样。
'''
learning_rate = 0.01
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784 # MNIST data input(img shape:28*28)

# tf Graph input(only pictures)
X = tf.placeholder("float",[None,n_input])

# hidden layer settings 784 -> 256 -> 128
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),

    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}

biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),

    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input]))
}

# Building the encoder
def encoder(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),
                                   biases['encoder_b1'])) # 第一层压缩成256
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),
                                   biases['encoder_b2'])) # 第二层压缩成128
    return layer_2

# Building the decoder
# 通常情况下，encoder用的是什么activation function,那么decoder用的也是一样的，一一对应
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))  # 第一层压缩成256
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))  # 第二层压缩成128

    return layer_2

# Constant model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op

# Target(Labels) are the input data
y_true = X

# Define loss and optimizer,minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variable
init = tf.initialize_all_variables()

# Lauch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        ## Loop over all batches
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size) # max(x)=1,min(x)=0
            _,c = sess.run([optimizer,cost],feed_dict={X:batch_xs})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:",'%04d'%(epoch + 1),
                    "cost=","{:.9f}".format(c))

    print("Optimizer Finished!")

    ## Applying encode and decode over test set
    encode_decode = sess.run(y_pred,feed_dict={X:mnist.test.images[:examples_to_show]})

    f,a = plt.subplots(2,10,figsize=(10,2))

    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    plt.show()