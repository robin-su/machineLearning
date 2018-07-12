# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    # 初始化权重
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))

    # 初始化偏移量
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    # 相乘 -> 预测的值
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    print(Wx_plus_b.shape)

    if activation_function is None: # 如果他是none，我们不做任何操作（线性）
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 创建等差数量-1到1,间隔300个,并且给x_data添加一个维度
# x_data是300*1的矩阵
x_data = np.linspace(-1,1,300)[:,np.newaxis] #np.newaxis实现增加列的维度;[np.newaxis,:]表示行增加维度
noise = np.random.normal(0,0.05,x_data.shape) # 随机使其满足标准正太分布
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# 输入x_data的数据
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu) #定义输入层
predition = add_layer(l1,10,1,activation_function=None) # 上一层的输出作为下一层的而输入

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                     reduction_indices=[1])) #reduce_sum:reduction_indices=[1]理解为pandas的按行求和

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 括号里面表示learning rate

# 初始化所有的变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig = plt.figure() # 生成一个图片框
ax = fig.add_subplot(1,1,1) # 1行1列第一个模块
ax.scatter(x_data,y_data) # 以点的形式画原始数据图
plt.ion()# 画完会使得整个程序停止，因此可以使用plt.ion()使得show之后不要停止
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50:
        # to see the step improvement
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        try:
            # 避免重复画线
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predition_value = sess.run(predition,feed_dict={xs:x_data})
        # 将predition_value的值以曲线的形式画上去
        lines = ax.plot(x_data,predition_value,'r-',lw=5)  #r- 表示使用红色的线,lw表示宽度
        plt.pause(0.1) # 暂停0.1秒再继续
