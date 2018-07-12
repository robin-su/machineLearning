# -*- coding:utf-8 -*-

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV

# 画出分类结果的决策边界
def plot_decision_boundary(pred_func):

    # 设定最大最小值，附加一点点边缘填充
    x_min,x_max = x[:,0].min() - .5,x[:,0].max() + .5
    y_min,y_max = x[:,1].min() - .5,x[:,1].max() + .5
    h = 0.01

    # 关于网格函数比较好的解释https://www.cnblogs.com/shenxiaolin/p/8854197.html
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    print(xx)
    print(yy)

    print("========================= After C_ ==============================")
    print(np.c_[xx.ravel(),yy.ravel()])

    # 用预测函数预测一下,关于c_ : https://blog.csdn.net/zhuzuwei/article/details/78022629
    z = pred_func(np.c_[xx.ravel(),yy.ravel()]) # reval（将多维数组降位一维）
    z = z.reshape(xx.shape)

    # 然后画出图
    plt.contourf(xx,yy,z,cmap=plt.cm.Spectral)
    plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Spectral)
    plt.show()




if __name__ == '__main__':
    np.random.seed(0)
    x, y = make_moons(200, noise=0.20)

    plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    # LogisticRegressionCV使用了交叉验证来选择正则化系数C
    clf = LogisticRegressionCV()
    clf.fit(x,y)

    plot_decision_boundary(lambda x:clf.predict(x))
    plt.title("Logistic Regression")
    plt.show()



