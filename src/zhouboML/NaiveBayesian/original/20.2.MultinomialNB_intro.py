#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# 多项式朴素贝叶斯
if __name__ == "__main__":
    np.random.seed(0)
    M = 20 # 20个样本
    N = 5  # 每个样本都是5维的,特征数量
    x = np.random.randint(2, size=(M, N))     # [low, high)
    # 去重，首先遍历x,然后将t拿出来做成一个tuple(确定python这个值是不会发生改变的),然后放入set中，set中的数据不可重复
    # 这个方法虽然很low，但是是一个放之四海而皆准的方法
    x = np.array(list(set([tuple(t) for t in x])))
    M = len(x)
    y = np.arange(M) # 认为每一行数据就是一个类别，一共M个类别
    print('样本个数：%d，特征数目：%d' % x.shape)
    print('样本：\n', x)
    # 动手：换成GaussianNB()试试预测结果？alpha多项式分布都要满足拉普拉斯平滑，alpha表示拉普拉斯平滑
    mnb = MultinomialNB(alpha=1)
    mnb.fit(x, y)
    y_hat = mnb.predict(x)
    print('预测类别：', y_hat)
    print('准确率：%.2f%%' % (100*np.mean(y_hat == y)))
    print('系统得分：', mnb.score(x, y))
    # from sklearn import metrics
    # print metrics.accuracy_score(y, y_hat) # 这个也可以算准确率
    err = y_hat != y
    for i, e in enumerate(err):
        if e:
            print(y[i], '：\t', x[i], '被认为与', x[y_hat[i]], '一个类别')
