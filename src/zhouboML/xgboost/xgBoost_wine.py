# -*- coding:utf-8 -*-

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    # print(acc)
    print(tip + '正确率：\t', float(acc.sum()) / a.size)



if __name__ == "__main__":
    data = np.loadtxt(r'D:\machineLearning\src\zhouboML\xgboost\data\12.wine.data', dtype=float, delimiter=',')
    y, x = np.split(data, (1,), axis=1) # 表示(1,)表示0到1列(不包含1)即第0列表示类别，其他列表示特征数据,axis=1表示以列划分
    # x = StandardScaler().fit_transform(x) # 归一化，均值为0，方差为1
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    # Logistic回归
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    show_accuracy(y_hat, y_test, 'Logistic回归 ')

    # XGBoost
    y_train[y_train == 3] = 0  # 将类别是3的都变成0，xgboost要求类别从0开始，因此把3的都设置成0
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    show_accuracy(y_hat, y_test, 'XGBoost ')



