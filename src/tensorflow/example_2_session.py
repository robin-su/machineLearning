# -*- coding:utf-8 -*-
import tensorflow as tf

# 构建一个一行两列的matrix
matrix1 = tf.constant([[3,3]])
# 构建一个两行一列的matrix
matrix2 = tf.constant([[2],
                      [2]])
product = tf.matmul(matrix1,matrix2) # matrix multiply 矩阵乘法(numpy中np.dot(m1,m2))

# method 1
sess = tf.Session()
# 执行下结构
# result = sess.run(product)
# print(result) # [[12]]
# sess.close()

# method 2 (这种方式，Session会自动关上)
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

