#_*_ coding:utf-8 _*_
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.fontconfig_pattern as fontPattern
import os

#构造数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

'''
    描述：计算inX附近的k个实例中哪一种类别最多，并将类别最多的那个类别作为inX所属类别
    k-近邻算法  这个比我自己实现的那个牛逼的感觉
    inX:要识别的实例（它到底属于哪个类型）
    dataSet: 表示训练集特征集合
    lebels: 表示标签集合
    k: 表示阈值k
'''
def classfy0(inX,dataSet,labels,k):
    dataSetsize = dataSet.shape[0]  # 计算特征集合的行数，即特征向量的个数
    diffMat = tile(inX, (dataSetsize, 1)) - dataSet # 求出inX和dataSet中每个向量对应的距离，并返回一个新的距离二维矩阵
    sqDifat = diffMat ** 2 # 计算 (x1 - x2) ^ 2
    sqlDistances = sqDifat.sum(axis=1) #(x1 - x2)^2 + (y1 - y2) ^ 2  axis=1 行相加
    distances = sqlDistances ** 0.05
    sortedDistances = distances.argsort() # 返回由小到大的排序的后的下标
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistances[i]] # 取出前k个距离最近的类别
        classCount[votelabel] = classCount.get(votelabel,0) + 1 # 非常优雅的写法,如果votelabel不存在返回0+1,否则返回值 +1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 解析文件
'''
    python中的try结构中都变量都作用域相当于try的作用域,因此不用提前定义，后面的代码也可以正常引用。
'''
def file2matrix(filename):
    try:
        with open(filename, 'r') as file:
            arrayLines = file.readlines()
            numberofLines = len(arrayLines) # 获取文件行数
            returnMat = zeros((numberofLines, 3)) # 构造一个arrayLines行3列的0矩阵，用于存储训练特征集
            classLabelVector = []  # 由于存储标签
            labelDict = {"didntLike":1,"smallDoses":2,"largeDoses":3}
            index = 0 # 标记第几行
            for line in arrayLines:
                line = line.strip() # 截取回车字符
                listFromLine = line.split('\t')
                returnMat[index,:] =  listFromLine[:3]
                classLabelVector.append(labelDict[listFromLine[-1]]) # 构建标签集合
                index += 1
            return returnMat,classLabelVector
    except Exception as e:
        print("Error reading file:%s" % e)

'''
    归一化特征值
        newData = (oldData-min)/(max-min)
    '''
def autoNorm(dataSet):
    minVal = dataSet.min(0) # 求出每一列的最小值
    maxVal = dataSet.max(0) # 求出每一列的最小值
    ranges = maxVal - minVal
    normData = zeros(shape(dataSet)) # 建立一个跟dataSet一样维度的0矩阵
    m = dataSet.shape[0] # 返回第一维的长度
    normData = dataSet - tile(minVal,(m,1))
    normData = normData / tile(ranges, (m, 1))
    return normData,ranges,minVal

# 计算错误率
def datingClassTest():
    hoRatio = 0.10
    # 解析文件 ETL
    featMat,labelMat = file2matrix(r'D:\machineLearning\src\KNN\MLRealBattle\datingTestSet.txt')
    # 归一化
    normMat,ranges,minVals = autoNorm(featMat)
    # 取出维度,也就是实例个数
    m = normMat.shape[0]
    # 计算测试集个数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        '''
            normMat[i,:] : 取出二维数组normMat中的第i行数组  --> 测试
            normMat[numTestVecs:m,:]: 取出二维数组中第numTestVecs到m-1行的数组
        '''
        classifierResult = classfy0(normMat[i,:],normMat[numTestVecs:m,:],labelMat[numTestVecs:m],3)
        print("the classfier came back with:%d,the real answer is:%d " % (classifierResult,labelMat[i]))
        if classifierResult != labelMat[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    return errorCount/float(numTestVecs)

# 构建完整的系统
def classfyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input('percentage of time spent palying video games?'))
    ffiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('D:\machineLearning\src\KNN\MLRealBattle\datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffiles,percentTats,iceCream])
    classifierResult = classfy0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:", resultList[classifierResult - 1])

# 手写识别系统像素转换成向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linstr = fr.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(linstr[j])
    return returnVect

# 手写识别系统构建
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir(r'D:\02_深度学习基础\machinelearninginaction\Ch02\digits\trainingDigits') # 列出dictory下的所有的文本文件
    m = len(trainingFileList) # 文件个数个数
    trainingMat = zeros((m,1024)) # 构建m行1024列矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i] # 取出第i个文件名
        fileStr = fileNameStr.split('.')[0]     #取出文件名，例如0_0.txt，则取出0_0
        classNumStr = int(fileStr.split('_')[0]) # 取出类别0_0 取出前面的0
        hwLabels.append(classNumStr) # 将类别添加到hwLabels中
        trainingMat[i,:] = img2vector(r'D:\02_深度学习基础\machinelearninginaction\Ch02\digits\trainingDigits/%s' % fileNameStr) #将文件转换成1*1024矩阵
    testFileList = os.listdir(r'D:\02_深度学习基础\machinelearninginaction\Ch02\digits\trainingDigits')   # 获取所有文本文件
    errorCount = 0.0
    mTest = len(testFileList) # 获取文本文件长度
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'D:\02_深度学习基础\machinelearninginaction\Ch02\digits\trainingDigits/%s' % fileNameStr)
        classifierResult = classfy0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


# 画图
def drawingDataAdvance(datingDataMat,datingLabels):
    plt.figure(figsize=(12, 6), dpi=80)
    axes = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    print
    'range(len(datingLabels)):'
    print
    range(len(datingLabels))
    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:  # 不喜欢
            type1_x.append(datingDataMat[i][0])
            type1_y.append(datingDataMat[i][1])

        if datingLabels[i] == 2:  # 魅力一般
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:  # 极具魅力
            print
            i, '：', datingLabels[i], ':', type(datingLabels[i])
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')

    plt.xlabel(u'Frequent Flyier Miles Earned Per Year')
    plt.ylabel(u'Percentage of Time Spent Playing Video Games')
    axes.legend((type1, type2, type3), (u'didntLike', u'smallDoses', u'largeDoses'), loc=2)
    plt.show()

if __name__ == '__main__':
    returnMat, classLabelVector = file2matrix('D:\machineLearning\src\KNN\MLRealBattle\datingTestSet.txt')
    drawingDataAdvance(returnMat,classLabelVector)
    # handwritingClassTest()