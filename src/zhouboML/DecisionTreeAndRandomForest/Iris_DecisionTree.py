#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
# iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'D:\machineLearning\src\zhouboML\data\8.iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1) # (4,)前4列是x,从第4列开始是y
    # 为了可视化，仅使用前两列特征
    x = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 划分训练集和测试集合也可以指定train_size=0.7
    #ss = StandardScaler()
    #ss = ss.fit(x_train)

    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    '''
		3、矩阵标准化 StandardScaler  ==> 无量纲化
		矩阵标准化的目的是，通过标准化处理，得到均值为0，标准差为1的服从标准正态分布的数据。（相对一维数据来说，也就是相对矩阵的每一列，数据的每一个维度），

		矩阵标准化使用如下公式完成：x^{'}=\frac{x-\mu }{\sigma } ，μ表示均值，σ表示标准差，（u和σ都可以看成是行向量形式，它们的每个元素分表表示矩阵每一列的均值和方差）可以看出矩阵中心化是矩阵标准化的一步，将中心化的矩阵除以标准差得到标准化矩阵。下面解释为什么要对矩阵进行标准化。

		在一些实际问题中，我们得到的样本数据都是多个维度的，即一个样本是用多个特征来表征的。比如在预测房价的问题中，影响房价y的因素有房子面积x_{1}、卧室数量x_{2}等，我们得到的样本数据就是(x_{1},x_{2})这样一些样本点，这里的x_{1}、x_{2}又被称为特征。很显然，这些特征的量纲和数值得量级都是不一样的，在预测房价时，如果直接使用原始的数据值，那么他们对房价的影响程度将是不一样的，而通过标准化处理，可以使得不同的特征具有相同的尺度（Scale）。这样，在使用梯度下降法学习参数的时候，不同特征对参数的影响程度就一样了。
	'''
    model = Pipeline([
        ('ss', StandardScaler()),#无量纲化。保证每一个特征的均值都是0，方差都是1。这样做往往可以提高分类效果
        ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3))]) # max_depth 指定树的最大深度,entropy表示使用的熵是ID3算法。C4.5是信息增益,CART是基尼系数
    # clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    model = model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)      # 测试数据

    # 保存
    # dot -Tpng -o 1.png 1.dot
    f = open('.\\iris_tree.dot', 'w')
    tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f) # model.get_params('DTC')['DTC']返回的是DecisionTreeClassifier(criterion='entropy', max_depth=3))分类器

    # 画图
    N, M = 100, 100  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围   x[:, 0].min()第0列的最小值
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围   x[：,1].min()第1列的最小值
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点 x1.flat表示将x1拉平，x2.flat将x2拉平。形成一个向量，以列的方式

    # # 无意义，只是为了凑另外两个维度
    # # 打开该注释前，确保注释掉x = x[:, :2]
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)  # 预测值
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w') # 背景设置成白色
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示 0，1，2三个类别对应的值用cm_light指定背景块颜色
    ''' 将测试数据画上去
    	x_test[:, 0] 表示测试数据的横坐标
    	x[:, 1] 表示测试数据的纵坐标
    	y_test.ravel():表示类别，ravel():拉平数组
    	edgecolors：指定图中圆点的边的颜色
    	s:指的是圆圈的平方是多少
    	cmap：指定圆圈的颜色
    	marker：使用什么形状
    '''
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=100, cmap=cm_dark, marker='o')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max) # x轴从x最小画到x1最大，去掉会出现留白
    plt.ylim(x2_min, x2_max)
    plt.grid(True) # 设置网格
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
    plt.show()

    # 训练集上的预测结果
    y_test = y_test.reshape(-1)
    result = (y_test_hat == y_test)   # True则预测正确，False则预测错误
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))

    # 过拟合：错误率
    depth = np.arange(1, 15) # 假定决策树的最小层数是一层，最大层数是14层
    err_list = [] # 错误率列表
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf = clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        err = 1 - np.mean(result) # 返回测试集的预测值与测试集的实际值偏差的平均值
        err_list.append(err)
        # print d, ' 准确度: %.2f%%' % (100 * err)
        print(d, ' 错误率：%.2f%%' % (100 * err))
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2) # 横轴depth深度，纵轴错误err_list,红色的o-,线宽为2
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()
