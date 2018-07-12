# -*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow as tf
tf.set_random_seed(1)

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1',shape=[1],dtype=tf.float32,initializer=initializer)
    var2 = tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)
    var21 = tf.Variable(name='var2',initial_value=[2.1],dtype=tf.float32)
    var22 = tf.Variable(name='var2',initial_value=[2.2],dtype=tf.float32)

with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3',shape=[1],dtype=tf.float32,initializer=initializer)
    var4 = tf.Variable(name='var4',initial_value=[4],dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4',initial_value=[4],dtype=tf.float32) # 虽然名字是一样的，但是很明显是两个变量

with tf.variable_scope("a_variable_reuse_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var5 = tf.get_variable(name='var5',shape=[1],dtype=tf.float32,initializer=initializer)
    scope.reuse_variables() # 告诉tensorflow，后面的变量是可以重复利用的
    var5_reuse = tf.get_variable(name='var5') #var5_reuse和var5是同一个变量

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # print(var1.name) # var1:0  # 若使用get_variable这种方式创建变量，则name_scope是无效的
    # print(sess.run(var1)) # [1.]
    # print(var2.name) # a_name_scope/var2:0
    # print(sess.run(var2)) #[2.]
    # print(var21.name) # a_name_scope/var2_1:0
    # print(sess.run(var21)) # [2.1]
    # print(var22.name) # a_name_scope/var2_2:0
    # print(sess.run(var22)) # [2.2]
    # print(var3.name)  # a_variable_scope/var3:0 使用variable_scope时，get_variable会被纳入variable_scope
    # print(sess.run(var3)) # [3.]
    # print(var4.name) #a_variable_scope/var4:0
    # print(sess.run(var4)) # [4.]
    # print(var4_reuse.name) # a_variable_scope/var4_1:0
    # print(sess.run(var4_reuse))  #[4.]

    print(var5.name) #a_variable_reuse_scope/var5:0
    print(sess.run(var5)) #[3.]
    print(var5_reuse.name) # a_variable_reuse_scope/var5:0
    print(sess.run(var5_reuse)) #[3.]