#_*_ coding:utf-8 _*_
from sklearn import svm
import numpy as np
import pylab as pl

np.random.seed(0)
# 随机生成一个20*2的矩阵
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# 标签,把前20个归类为0，把后20个归类为1
Y= [0] * 20 + [1] * 20
# 实例化SVM,注意kernel表示核函数，这里是线性可分的，因此使用
clf = svm.SVC(kernel='linear')
# 建立模型
clf.fit(X,Y)

'''
    分割超平面方程为：w0x + w1y + w3 = 0
    我们要找出他的点斜式： y = -[w0/w1]x + [w3/w1]
'''
w = clf.coef_[0] # w = [w0,w1]
a = - w[0]/w[1] # 斜率
xx = np.linspace(-5,5) #  创建 -5到5之间的等差数列，个数为50个，默认50，可以指定
yy = a * xx - (clf.intercept_[0]) / w[1] # 点斜式 intercept_[0]取到的是w3


# 画出与支持向量机相切的两条直线
b = clf.support_vectors_[0] # 取出第一个支持向量
yy_down = a * xx + (b[1] - a * b[0]) #  b[1] - a * b[0]是使用第一个支持向量计算出来的截距

b = clf.support_vectors_[-1] # 取出最后一个支持向量
yy_up = a * xx + (b[1] - a * b[0]) #  b[1] - a * b[0]是使用最后一个支持向量计算出来的截距


# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
