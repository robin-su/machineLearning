import operator

dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
dict1 = dict({1:'c', 2:'a', 3:'b'})

def testList():
    for row in dataSet:
        tmp = row[:2]  # 这种写法等价于row[0:2]
        print(tmp)

def testDict():
    sortedDict1 = sorted(dict1.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedDict1)

testDict()