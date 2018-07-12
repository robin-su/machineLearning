# -*- coding:utf-8 -*-

import sys
import re
import numpy as np
from sklearn.externals import joblib
import csv
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# 处理域名的最小长度
MIN_LEN = 10

# 随机程度
random_state = 170

def load_alexa(filename):
    domain_list=[]
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        domain = row[1]
        if len(domain) >= MIN_LEN:
            domain_list.append(domain)
    return domain_list

def load_dga(filename):
    domain_list=[]
    with open(filename) as f:
        for line in f:
            domain = line.split(",")[0] # 获取域名
            if len(domain) >= MIN_LEN:
                domain_list.append(domain)
    return domain_list



# 高斯朴素贝叶斯 -> 聚类分析 检测DGA

'''
    原理：
    1.提取1000个正常的常用域名,1000个DGA动态生成的域名
    2.使用2-gram模型进行特征提取
    3.使用GaussianNB（高斯朴素贝叶斯）进行分类
    4.计算AUC
'''
def kmeans_dga():
    x1_domain_list = load_alexa("C:/Users/robin/Desktop/resume/1book-master/data/dga/top-100.csv")
    x2_domain_list = load_alexa("C:/Users/robin/Desktop/resume/1book-master/data/dga/dga-cryptolocke-50.txt")
    x3_domain_list = load_dga("C:/Users/robin/Desktop/resume/1book-master/data/dga/dga-post-tovar-goz-50.txt")

    x_domain_list=np.concatenate((x1_domain_list,x2_domain_list,x3_domain_list))


    y1 = [0] * len(x1_domain_list)
    y2 = [1] * len(x2_domain_list)
    y3 = [1] * len(x3_domain_list)

    y = np.concatenate((y1,y2,y3))

    # 建立词频矩阵，注意这里采用的2 gram的方式
    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                         token_pattern=r"\w", min_df=1)

    x = cv.fit_transform(x_domain_list).toarray()

    # 使用KMeans++ 进行聚类分析
    model = KMeans(n_clusters=2,init='k-means++')
    # model = KMeans(n_clusters=2, random_state=random_state)
    y_pred = model.fit_predict(x)

    #同一性：每个群集只包含单个类的成员。
    h = metrics.homogeneity_score(y,y_pred)
    print(u'同一性：',h)
    # 完整性：给定类的所有成员都分配给同一个群集。
    c = metrics.completeness_score(y,y_pred)
    print(u'完整性：',c)
    # 调和平均数:均一性和完整性的加权平均
    v = metrics.v_measure_score(y,y_pred)
    print(u'V-Measure：', v)



    # 使用TSNE进行降低维度
    tsne = TSNE(learning_rate=100)
    x = tsne.fit_transform(x)

    for i, label in enumerate(x):
        # print label
        x1, x2 = x[i]
        if y_pred[i] == 1:
            plt.scatter(x1, x2, marker='o')
        else:
            plt.scatter(x1, x2, marker='x')
        # plt.annotate(label,xy=(x1,x2),xytext=(x1,x2))

    plt.show()

def nb_dga():
    x1_domain_list = load_alexa("C:/Users/robin/Desktop/resume/1book-master/data/top-1000.csv")
    x2_domain_list = load_dga("C:/Users/robin/Desktop/resume/1book-master/data/dga-cryptolocke-1000.txt")
    x3_domain_list = load_dga("C:/Users/robin/Desktop/resume/1book-master/data/dga-post-tovar-goz-1000.txt")

    x_domain_list = np.concatenate((x1_domain_list, x2_domain_list, x3_domain_list))
    for i in range(1000,1030):
        print(x_domain_list[i])

    y1 = [0] * len(x1_domain_list)
    y2 = [1] * len(x2_domain_list)
    y3 = [2] * len(x3_domain_list)

    y = np.concatenate((y1, y2, y3))


    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                         token_pattern=r"\w", min_df=1)

    x = cv.fit_transform(x_domain_list).toarray()

    print(cv.vocabulary_)

    clf = GaussianNB()
    print(cross_val_score(clf, x, y, n_jobs=-1,cv=3))


if __name__ == '__main__':
    # kmeans_dga()
    nb_dga() #0.9387746240114905