# -*- coding: utf-8 -*- 

import numpy as np
from pandas import Series, DataFrame

print('Series的索引，默认数字索引可以工作。')
obj = Series(np.arange(4.), index = ['a', 'b', 'c', 'd'])
print(obj['b']) 
print(obj[3]) # series会有一个默认的传统数组一样的索引在那里与他匹配
print(obj[[1, 3]]) #花式索引
print(obj[obj < 2]) 
print

print('Series的数组切片')
print(obj['b':'c'])  # 闭区间，对于非数字索引都是闭区间，因为没法算开区间
obj['b':'c'] = 5
print(obj)
print

print('DataFrame的索引')
data = DataFrame(np.arange(16).reshape((4, 4)),
                  index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                  columns = ['one', 'two', 'three', 'four'])
print(data)
print(data['two']) # 打印列
print(data[['three', 'one']])
print(data[:2])
print(data.ix['Colorado', ['two', 'three']]) # 指定索引和列,ix的第一个参数代表行，第二个参数代表列
print(data.ix[['Colorado', 'Utah'], [3, 0, 1]]) # 对于每个非数字索引都有一个数字索引与他对应
print(data.ix[2])  # 打印第2行（从0开始）
print(data.ix[:'Utah', 'two']) # 从开始到Utah，第2列。对于非数字索引这里一定是闭区间
print

print('根据条件选择')
print(data[data.three > 5])  #打印第三列中，元素大于5的
print(data < 5)  # 打印True或者False，看每一个元素，若小于5返回true，若大于5返回false
data[data < 5] = 0  # 将data<5的值全部设置成0
print(data)
