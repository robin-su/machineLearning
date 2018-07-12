# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

def show_accuracy(a,b,tip):
    acc = a.ravel() == b.ravel()
    print(tip + "正确率：",np.mean(acc))

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    path = r'D:\machineLearning\src\zhouboML\xgboost\data\8.iris.data'
    data = np.loadtxt(path,dtype=float,delimiter=',',converters={4:iris_type})
    x,y = np.split(data,(4,),axis=1)
    x = x[:, :2]
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.6)

    # 使用交叉验证确定超参数
    model = svm.SVC(kernel='rbf')
    c_can = np.logspace(-2,2,10)
    gamma_can = np.logspace(-2,2,10)
    clf = GridSearchCV(model,param_grid={'C':c_can,'gamma':gamma_can},cv=5)
    clf.fit(x_train,y_train.ravel())
    print('验证参数：\n', clf.best_params_)

    # 计算训练数据的准确率
    print(clf.score(x_train,y_train))
    y_hat = clf.predict(x_train)
    show_accuracy(y_train,y_hat,"训练集")
    print(clf.score(x_test,y_test))
    y_hat = clf.predict(x_test)
    show_accuracy(y_test,y_hat,"测试集")

    # 画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    # 查看每个样本到分隔超平面的距离
    Z = clf.decision_function(grid_test)
    print(Z)

    grid_hat = clf.predict(grid_test)
    print(grid_hat)

    grid_hat = grid_hat.reshape(x1.shape)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 分类块的颜色
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    # 样本点的颜色
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=50, cmap=cm_dark)  # 样本
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
    plt.grid()
    plt.show()




