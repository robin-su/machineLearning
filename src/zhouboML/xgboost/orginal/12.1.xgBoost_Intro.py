# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np

# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw


# 定义f: theta * x
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
    # 读取数据
    data_train = xgb.DMatrix('D:\\machineLearning\\src\zhouboML\\xgboost\\data\\12.agaricus_train.txt')
    data_test = xgb.DMatrix('D:\\machineLearning\\src\zhouboML\\xgboost\\data\\12.agaricus_test.txt')

    # 设置参数
    '''
        max_depth: 每一棵树最大的深度是多少
        eta : 给一个防止过你和的参数， v = 1即为原始模型，推荐选择v<0.1的小学习率。过小的学习率会造成计算次数增多
        silent : 1 表示沉默意思是说，这棵树的建立过程不要显示出来，0表示将树的建立过程显示出来
        'binary:logistic' 表示而分类问题，若是多分类，可以选择softmax  logistic和logitraw的区别，logistic他的输出值
        都是大于0小于1的数（(y_hat > 0.5)表示1），logitraw他的输出是-∞到+∞的一个数，此时(y_hat > 0)表示1，
        
    '''

    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logitraw'} # logitraw
        # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')] # 指定测速数据和训练数据
    n_round = 3 # 树的数量
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    '''
        使用自定义损失函数：
        obj=log_reg 指定自定义目标函数是log_reg
        feval=error_rate 指定自定义损失函数error_rate
    '''
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label() # 获取测试数据的标记
    print(y_hat)
    print(y)
    error = sum(y != (y_hat > 0)) # 若y_hat 大于0 则判定为1.否则判定为0
    error_rate = float(error) / len(y_hat)
    print('样本总数：\t', len(y_hat))
    print('错误数目：\t%4d' % error)
    print('错误率：\t%.5f%%' % (100*error_rate))
