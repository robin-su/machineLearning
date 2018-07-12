# -*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow as tf
tf.set_random_seed(1)

'''
    reuse应用场景：在training时创建的神经网络和test时创建的神经网络结构
                是不一样的，但是结构不一样又导致输入输出不一样，但是我又
                希望在traing rnn和test rnn中他们的参数是一样的。这个
                时候我们就需要使用到reuse_variables
    
    例子有问题，只是为了说明
'''
class TrainConfig:
    batch_size = 20
    time_steps = 20
    input_size = 10
    output_size = 2
    cell_size = 11
    learning_rate = 0.01


class TestConfig(TrainConfig):
    time_steps = 1

class RNN(object):
    def __init__(self,config):
        self._batch_size = config.batch_size
        self._time_steps = config.time_steps
        self._input_size = config.input_size
        self._cell_size = config.cell_size
        self._lr = config.learning_rate
        self._built_RNN()

    def _built_RNN(self):
        with tf.variable_scope('inputs'):
            self._xs = tf.placeholder(tf.float32,[self._batch_size])
            self._ys = tf.placeholder(tf.float32, [self._batch_size])
        with tf.name_scope('RNN'):
            with tf.variable_scope('input_layer'):
                l_in_x = tf.reshape(self._xs,[-1,self._input_size])
                Wi = self._weight_variable([self._input_size,self])
                print(Wi.name)

if __name__ == '__main__':
    train_config = TrainConfig()
    test_config = TestConfig()

    # 这里没共享变量
    # with tf.variable_scope('train_rnn'):
    #     train_rnn1 = RNN(train_config)
    # with tf.variable_scope("test_rnn"):
    #     test_rnn1 = RNN(test_config)

    #
    with tf.variable_scope('rnn') as scope:
        sess = tf.Session()
        train_rnn2 = RNN(train_config)
        scope.reuse_variables()
        test_rnn2 = RNN(test_config)
        sess.run(tf.initialize_all_variables())