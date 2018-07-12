#_*_ coding:utf-8 _*_
import csv
import operator
import random

import math

# 加载数据
def loadDataset(filename,split,trainingSet = [],testSet = []):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        # 划分训练集和测试集
        for x in range(len(dataset)-1):
            for y in range(len(dataset[0])-1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# 计算欧式距离(注意这里是两个向量)
def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

# 找出距离最小的前k个实例
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# 多数表决发决定实例属于哪个类别
def getResponse(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        reponse = neighbors[x][-1]
        if reponse in classVote.keys():
            classVote[reponse] += 1
        else:
            classVote[reponse] = 1
    sortedVotes = sorted(classVote.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def getccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100


if __name__ == '__main__':
    trainingset = []
    testset = []
    split = 0.67
    loadDataset(r"D:\machineLearning\src\KNN\irisdata.txt",0.67,trainingset,testset)
    print("Train set: " + repr(len(trainingset)))
    print("Test set: " + repr(len(testset)))
    predictions = []
    k = 3
    for x in range(len(testset)):
        neighbor = getNeighbors(trainingset,testset[x],k)
        result = getResponse(neighbor)
        predictions.append(result)
        print('>predicted=' + repr(result) + ', actual=' + repr(testset[x][-1]))
    print('predictions: ' + repr(predictions))
    accuracy = getccuracy(testset,predictions)
    print('Accuracy: ' + repr(accuracy) + "%")




