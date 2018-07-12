# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

'''
    决策树回归实例
'''
if __name__ == "__main__":
    N = 100 # 100个样本
    # [-3,3) np.random.rand(N)随机取100个满足均匀分布的0~1的点，*6则返回0~6,-3则返回-3到3
    x = np.random.rand(N) * 6 - 3
    x.sort() # 从小到大排序
    y = np.sin(x) + np.random.randn(N) * 0.05 # np.random.randn(N) * 0.05表示噪声 均值是0.标准差是0.05的100个样本累加到np.sin(x)中
    print(y)
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的 只有一个特征x
    print(x)

    reg = DecisionTreeRegressor(criterion='mse', max_depth=9)  # 做回归的时候，使用的不是熵或者基尼系数，而是使用均方误差mse
    dt = reg.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = dt.predict(x_test)
    plt.plot(x, y, 'r*', linewidth=2, label='Actual')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # 比较决策树的深度影响
    depth = [2, 4, 6, 8, 10]
    clr = 'rgbmy'
    reg = [DecisionTreeRegressor(criterion='mse', max_depth=depth[0]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[1]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[2]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[3]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[4])]

    plt.plot(x, y, 'k^', linewidth=2, label='Actual')
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    for i, r in enumerate(reg):
        dt = r.fit(x, y)
        y_hat = dt.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=clr[i], linewidth=2, label='Depth=%d' % depth[i])
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
