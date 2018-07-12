# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    acccuracy = tf.reduce_mean(tf.cast(correct_prediction ,tf.float32))
    result = sess.run(acccuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result


def weight_variable(shape):

    '''
        tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，
        stddev是标准差。这个函数产生正太分布，均值和标准差自己设定
    '''
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


# 定义卷积神经网络层
def conv2d(x,W):
    # 在tensorflow中strides步长是一个4维的列表
    '''
    strides[0]=1，表示在 batch 维度上移动为 1，指不跳过任何一个样本，每一个样本都会进行运算
    strides[1] = 1，表示在高度上移动步长为1，这个可以自己设定，根据网络的结构合理调节
    strides[2] = 1，表示在宽度上的移动步长为1，这个可以自己设定，根据网络的结构合理调节
    strides[3] = 1，表示在 channels 维度上移动为 1，指不跳过任何一个颜色通道，每一个通道都会进行运算
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') # 一般strides[0]=strides[3]=1

def max_pool_2x2(x):
    '''
        第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
        因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    '''
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

'''
    将xs的shape变成[-1,28,28,1]：如果shape的一个分量是特殊值-1，则计算该维度的大小，以使总大小保持不变。
    特别地情况为，一个[-1]维的shape变平成1维。至多能有一个shape的分量可以是-1。
    shape[0]=-1：表示batch即样本数量，这个维度由后面的维度计算得到
    shape[1]=28: 图片的宽度为28
    shape[2]=28: 图片的高度为28
    shape[3]=1:  表示图片的channel,本例中是黑白图片因此是1，若是彩色的则为3
'''
x_image = tf.reshape(xs,[-1,28,28,1])
print(x_image.shape) # [n_samples,28,28,1]

## conv1 layer ##
# 表示filter大小是5*5,使用32个filter进行卷积，由于是黑白的所以输入通道是1.
W_conv1 = weight_variable([5,5,1,32]) # patch 5*5,in size 1,out size 32
#每一个输出通道都有一个偏置量
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1) # ouput size 由于stride = 2,则28/2=14，因此其维度为 14 * 14 * 32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # 此时通过经过layer1已经变成了32，这里再使用64个filter
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) # 14 * 14 * 64
h_pool2 = max_pool_2x2(h_conv2) # 7 * 7 * 64

## func1 layer ##
#图片尺寸减小到7*7，加入一个有1024个神经元的全连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#将最后的池化层输出张量reshape成一维向量
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
"""使用Dropout减少过拟合"""
#使用placeholder占位符来表示神经元的输出在dropout中保持不变的概率
#在训练的过程中启用dropout，在测试过程中关闭dropout
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## func2 layer 输出层 ##
W_fc2 = weight_variable([1024,10]) # 10 代表输出的是0,1,2,...,9
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1])) # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))




