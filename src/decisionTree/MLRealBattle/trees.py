#_*_ coding:utf-8 _*_
import operator
from math import log
import matplotlib.pyplot as plt

# 构建数据集
def createDateSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    label = ['no surfacing','flippers']
    return dataSet,label


# 计算数据集的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 获取数据集长度
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 开始计算熵 plogp
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prop = float(labelCounts[key])/numEntries
        shannonEnt -= prop * log(prop,2)
    return shannonEnt


# 按照给定的特征划分数据集,取出每个特征的第axis个属性的值为value的特征，并将第axis后的属性形成一个新的list
# 该算法为计算每个特征的信息增益做准备
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 计算每个特征的信息增益，并返回信息增益最大的那个特征
def chooseBestFeatureToSplit(dataSet):
    beseEntropy = calcShannonEnt(dataSet) # 数据集合的信息增益
    numFeatures =  len(dataSet[0]) - 1 # 获取特征的个数
    baseInfoGain = 0.0; bastFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # 类别标签组成的集合 （去除相同的元素）
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # 或找出dataSet中第i个特征值为value的数据集合
            prob = len(subDataSet) / float(len(dataSet)) # 计算频率（拟合概率）
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = beseEntropy - newEntropy # 计算每种特征向量的信息增益
        if(infoGain > baseInfoGain): # 取出最大的信息增益所对应的特征向量说对应的索引地址
            baseInfoGain = infoGain
            bastFeature = i
    return bastFeature

# 多数表决筛选出类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList.keys():
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 构建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] # ['yes', 'yes', 'no', 'no', 'no']
    if classList.count(classList[0]) == len(classList): # 第一种情况，当类别完全相同时，停止划分
        return classList[0]
    if len(dataSet[0]) == 1: # 当只有一个特征的时候，遍历完所有实例返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 筛选出信息增益最大的特征
    bestFeatLabl = labels[bestFeat]
    myTree = {bestFeatLabl : {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabl][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


dataSet, label = createDateSet()
# ent = calcShannonEnt(dataSet)
# print(ent)

# data_set = splitDataSet(dataSet, 1, 1)
# print(data_set)

# bestFeature = chooseBestFeatureToSplit(dataSet)
# print(bestFeature)

myTree = createTree(dataSet, label)
# print(myTree) #{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

'''
    绘制决策树
    nodeTxt：节点的文字标注,
    centerPt：节点中心位置,
    parentPt：箭头起点位置（上一节点位置）,
    nodeType：节点属性 
'''
decisionNode = dict(boxstyle='sawtooth',fc = '0.8') #boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc控制的注解框内的颜色深度
leafNode = dict(boxstyle = 'round4',fc = '0.8')
arrow_args = dict(arrowstyle='<-')

'''
     ax.annotate(node_txt, xy=parent_ptr, xycoords='axes fraction',  
                xytext=center_ptr, textcoords='axes fraction',  
                va="center", ha="center", bbox=node_type, arrowprops=arrow_args)  
                
     在该示例中，xy（箭头尖端）和xytext位置（文本位置）都以数据坐标为单位。 有多种可以选择的其他坐标系 - 你可以使用xycoords和textcoords以及
     下列字符串之一（默认为data）指定xy和xytext的坐标系。
        -----------------------------------------------------------  
        | 参数              | 坐标系                             |   
        -----------------------------------------------------------  
        | 'figure points'   | 距离图形左下角的点数量               |   
        | 'figure pixels'   | 距离图形左下角的像素数量             |   
        | 'figure fraction' | 0,0 是图形左下角，1,1 是右上角       |   
        | 'axes points'     | 距离轴域左下角的点数量               |   
        | 'axes pixels'     | 距离轴域左下角的像素数量             |     
        | 'axes fraction'   | 0,0 是轴域左下角，1,1 是右上角       |   
        | 'data'            | 使用轴域数据坐标系                  |  
'''
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va='center',ha='center',
                            bbox=nodeType,
                            arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

createPlot()



