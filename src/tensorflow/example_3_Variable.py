# -*- coding:utf-8 -*-

import tensorflow as tf

# 定义一个变量,初始值为0,名字为counter
state = tf.Variable(0,name='counter')
#print(state.name) #counter:0

one = tf.constant(1) # 常量

new_value = tf.add(state,one)
update = tf.assign(state,new_value) # 将new_value这个变量加载到state这个变量上面

# 在tensorflow中如果你设定了一些变量，那接下来的这一步就是最重要的一步
# 初始化所有的变量
init = tf.initialize_all_variables() # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))