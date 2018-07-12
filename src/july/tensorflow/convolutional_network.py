# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("E:\自学\04-深度学习课程\11月深度学习班\train_data",one_hot=True)

# 超参数
learning_rate = 0.001
training_iters = 200000 # 总共迭代的次数
batch_size = 128 # 每次迭代多少张图片
display_step = 10 # 展示

# 神经网络参数设定
n_input = 784 # 输入维度
n_classes = 10 # 分类个数
dropout = 0.75 # dropout

# tf 计算图输入
x = tf.placeholder(tf.float32,[None,n_input]) # n_input列，行不定
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)  # 有多少会保持活性

# 卷积层
def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME') #https://blog.csdn.net/lanchunhui/article/details/61615714
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x) # 包含着激励层

# 池化层
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1])

# 创建模型
def conv_net(x,weights,biases,droupout):
    x = tf.reshape(x,shape=[-1,28,28,1]) #https://www.jianshu.com/p/689b5b5b46d8

    # Convolution Layer
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1,k=2)

    # Convolution Layer
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2,k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(tf.matmul(fc1,weights['wd1']),biases['bd1']))
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1,dropout)

    # Output,class prediction
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])

    return out




weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),

    # 5x5 conv, 32 inputs, 64 outputs
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),

    # 1024 inputs, 10 outputs (class prediction)
    'out':tf.Variable(tf.random_normal([1024,n_classes]))
}

biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x,weights,biases,keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))  # 损失函数
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))

