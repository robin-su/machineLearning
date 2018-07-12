# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

'''
    序列模型的时候，一行一行的输入，每一行都是28个像素，一共28行，所以下面也是28
'''
n_inputs = 28 # MNIST data input(img shape:28*28) ,
n_steps = 28 # time steps
n_hidden_units = 128 # neurons in hidden layer 自己设定的
n_class = 10 # MNIST classes(0~9 digits)

# tf Graph input
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_class])

# Define weights
weights = {
  # (28,128)
  'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    # (128,10)
  'out':tf.Variable(tf.random_normal([n_hidden_units,n_class]))
}

biases = {
    #(128,)
  'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    #(10,)
  'out':tf.Variable(tf.constant(0.1,shape=[n_class,]))
}

def RNN(X,weights,biases):
    # hidden layer for input to cell
    ################################
    # X (128 batch,28 steps,28 inputs)
    # ==> (128*28,28 inputs)
    X = tf.reshape(X,[-1,n_inputs]) # n_inputs = 28

    # X_in ==> (128batch*28 steps,128 hidden)
    X_in = tf.matmul(X,weights['in']) + biases['in'] # 输入的x
    # X_in ==> (128batch,28 steps,128 hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units]) # 将x转换成3维的


    # cell
    ################################
    # n_hidden_units 表示有多少个隐藏测
    '''
        n_hidden_units表示神经元的个数，forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息。
        如果等于0，就都忘记。state_is_tuple默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示。
        那么当state_is_tuple=True的时候，state是元组形式，state=(c,h)
    '''
    # forget_bias = 1.0 因为初始的时候，不想忘记前面的东西，所以设置成1
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    # 参数初始化,rnn_cell.RNNCell.zero_stat
    # 初始化成全部都是0的state
    _init_state = lstm_cell.zero_state(batch_size,tf.float32)

    # dynamic_rnn 比 rnn效果要好
    # time_major=false 因为X_in(128batch,28 steps,128 hidden)中steps位于第二个，所以不是主要维度是次要维度，所以是false
    # 若是(28 steps,128batch,128 hidden) 则 time_major=True
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

    # hidden layer for output as the final results
    ################################
    # state包含两个(c_state,m_state)
    results = tf.matmul(states[1],weights['out']) + biases['out']

    return results

pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size) # 每次提取mnist的下一个batch进行处理
        '''
            由于每个batch提取的数据batch_xs都是一串数据，所以要将其reshape
            reshape[0]:表示每一批的数据有多少即batch_size个
            reshape[1]:表示28行
            reshape[1]:表示28列
        '''
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        '''
            run的参数里前面是操作的列表，后面依赖的数据统一放在feed_dict中，
            这样sess.run()返回的不是tensor对象，而是numpy的ndarray，处理起来就会比较方便了
        '''
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys,
        })

        if step % 20 == 0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys,
            }))
        step += 1

