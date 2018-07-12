#_*_ coding:utf-8 _*_
from sklearn import svm

x = [[2,0],[1,1],[2,3]] # 空间中的实例
y = [0,0,1]  # 标记与上面中实例对应

clf = svm.SVC(kernel='linear') # kernel='linear'代表线性的核函数
clf.fit(x,y)

print(clf)

# 打印支持向量
print(clf.support_vectors_)

# 打印支持向量在训练集中的索引
print(clf.support_)

# 打印支持向量的个数,针对两个类，每个类找出了一个支持向量
print(clf.n_support_)

# 预测实验
print(clf.predict([2,0])) #[0]
