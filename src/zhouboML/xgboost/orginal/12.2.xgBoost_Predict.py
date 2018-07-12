# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = r'D:\machineLearning\src\zhouboML\xgboost\data\8.iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_train) # 将训练数据他的标记组装成DMatrix形成训练集
    data_test = xgb.DMatrix(x_test, label=y_test) # 将测试数据和他的标记组装成DMatrix形成测试集
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    '''
        'objective': 'multi:softmax' 表示多分类问题
        num_class：3  表示有3个类别
    '''
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}

    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(data_test)
    result = y_test.reshape(1, -1) == y_hat
    print('正确率:\t', float(np.sum(result)) / len(y_hat))
    print('END.....\n')
