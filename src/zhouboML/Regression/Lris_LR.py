# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = u'D:\machineLearning\src\zhouboML\Regression\data\8.iris.data'
    df = pd.read_csv(path,header=0)
    x = df.values[:,:-1] # 读取从第一列到倒数第二列的数据
    y = df.values[:,-1] # 最后一列作为标签
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    y = le.transform(y)

    # 为了可视化，仅使用前两列特征
    x = x[:,:2]

    # 等价形式
    lr = Pipeline([('sc',StandardScaler())
                      ,('clf',LogisticRegression())]) # 本质是softmax回归
    lr.fit(x,y.ravel()) # 表示一列，fit函数要求的是一行，所以使用ravel进装置

    # 画图
    N,M = 500,500 # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min,x1_max,N)
    t2 = np.linspace(x2_min,x2_max,N)
    x1,x2 = np.meshgrid(t1,t2)  # 生成网格采样点
    '''
        测试点,axis=1表示按列堆加 flat：使平坦，在编程上就对应着二维变一维。
        flatten() 是函数调用，可以指定平坦化的参数。
    '''
    x_test = np.stack((x1.flat,x2.flat),axis=1)

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = lr.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.savefig('2.png')
    plt.show()

    # 训练集上的预测结果
    y_hat = lr.predict(x)
    y = y.reshape(-1)
    result = y_hat == y
    print(y_hat)
    print(result)
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))