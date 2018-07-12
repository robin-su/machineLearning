import csv
from sklearn import preprocessing, tree

from sklearn.feature_extraction import DictVectorizer

featureList = []  # 特征向量容器  --> 注意这可是特征和特征值组成的dict所组成的List
labelList = [] # 标签容器
data = [] # 原始数据集
header = []

try:
    with open('D:\machineLearning\src\decisionTree\AllElectronics.csv','r') as allElectronics:
        reader = csv.reader(allElectronics)
        header = next(reader) # ['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']
        data = [row for row in reader] # 使用列表解析
except csv.Error as e:
        print("Error reading CSV file at line %s: %s" % (reader.line_num,e))

# row ['1', 'youth', 'high', 'no', 'fair', 'no']

# 将featureList转换成：featureList = [{k1,v1},{k2,v2}]
for row in data:
    labelList.append(row[len(row)-1]) #数据集中的最有一行为标签
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[header[i]] = row[i]
    featureList.append(rowDict)

# 将featureList转换成数值类型键值矩阵
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX:" + str(dummyX))
print(vec.get_feature_names())

# 将labelList转换成数值类型向量
lb = preprocessing.LabelBinarizer()
dummY = lb.fit_transform(labelList)
print("dummyY: " + str(dummY))

# 构建决策树
clf = tree.DecisionTreeClassifier(criterion='entropy') # 使用熵的度量，也就是使用ID3算法
clf = clf.fit(dummyX,dummY)
print("clf: " + str(clf))

#画出决策树
# with open('allElectronicInformationGainOri.dot','w') as f:
#     f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

# 预测实验
oneRow = dummyX[0,:]

newRowX = oneRow

newRowX[0] = 1 # youth
newRowX[2] = 0 # 高收入
print("newRowX: " + str(newRowX))

willBePredict = []
willBePredict.append(newRowX)

predictedY = clf.predict(willBePredict)
print("predictedY: " + str(predictedY[0]))  # 预测会购买
