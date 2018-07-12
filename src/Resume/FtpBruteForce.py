# -*- coding:utf-8 -*-

import re
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn import cross_validation
from sklearn.model_selection import cross_val_score

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
import numpy as np

'''
    思路：1)通过ids获取系统的调用序列（其实就是一个序号记录在sys_call_table[]中）
         2)将样本分成正常的，和暴力破解的样本
         3)将训练集转换成词频矩阵
         4)使用随机森林进行训练和10折交叉验证进行分析
'''
def load_one_flle(filename):
    x=[]
    with open(filename) as f:
        line=f.readline()
        line=line.strip('\n')
    return line

'''
    加载正常的样本
'''
def load_adfa_training_files(rootdir):
    x=[]
    y=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            x.append(load_one_flle(path))
            y.append(0)

    print(x)
    print(y)

    return x,y

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

'''
    加载异常的样本
'''
def load_adfa_hydra_ftp_files(rootdir):
    x = []
    y = []
    allfile = dirlist(rootdir,[])
    for file in allfile:
        if re.match(r"C:/Users/robin/Desktop/resume/1book-master/data/ADFA-LD/Attack_Data_Master/Hydra_FTP_\d+/UAD-Hydra-FTP*",file):
            x.append(load_one_flle(file))
            y.append(1)
    return x,y

'''
1) max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:

max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
max_df = 25 means "ignore terms that appear in more than 25 documents".
The default max_df is 1.0, which means "ignore terms that appear in more than 100% of the documents". Thus, the default setting does not ignore any terms.

2) min_df is used for removing terms that appear too infrequently. For example:

min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
min_df = 5 means "ignore terms that appear in less than 5 documents".
The default min_df is 1, which means "ignore terms that appear in less than 1 document". Thus, the default setting does not ignore any terms.
'''
if __name__ == '__main__':
    x1,y1 = load_adfa_training_files("C:/Users/robin/Desktop/resume/1book-master/data/ADFA-LD/Training_Data_Master/")
    x2,y2 = load_adfa_hydra_ftp_files("C:/Users/robin/Desktop/resume/1book-master/data/ADFA-LD/Training_Data_Master/")

    x = x1 + x2
    y = y1 + y2
    print("before \n" + str(x))
    #
    vectorizer = CountVectorizer(min_df=1)
    x = vectorizer.fit_transform(x)
    print(vectorizer.vocabulary_)
    # 将训练集合X转换成词频矩阵
    x = x.toarray()
    print("after \n" + str(x))
    # 决策树
    clf1 = tree.DecisionTreeClassifier()
    '''
        该模块可以并行构建多棵树，以及并行进行预测，通过n_jobs参数来指定。如果n_jobs=k，则计算被划分为k个job，
        并运行在k核上。如果n_jobs=-1，那么机器上所有的核都会被使用。注意，由于进程间通信的开销，加速效果并不会
        是线性的（job数k不会提升k倍）。通过构建大量的树，比起单棵树所需的时间，性能也能得到很大提升。
        （比如：在大数据集上）
    '''
    score = cross_val_score(clf1,x,y,n_jobs=-1,cv=10)

    print(np.mean(score))
    # 随机森林
    clf2 = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=2, random_state=0)
    score = cross_val_score(clf2,x,y,n_jobs=-1,cv=10)
    print(np.mean(score))