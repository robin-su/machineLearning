# -*- coding: utf-8 -*- 

import numpy as np
from pandas import Series, DataFrame, MultiIndex

print('Series的层次索引')
data = Series(np.random.randn(10),
              index = [['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                       [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
print(data)
print(data.index)
print(data.b)
print(data['b':'c'])
print(data[:2]) #数字索引，就不算一级索引和二级索引了，只算前两行
print(data.unstack()) # 索引包括列的名字
print(data.unstack().stack()) #stack将列名放到索引中,使得 df 变成了二重索引
print

print('DataFrame的层次索引')
frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns = [['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
print(frame)
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
print(frame)
print(frame.ix['a', 1])
print(frame.ix['a', 2]['Colorado'])
print(frame.ix['a', 2]['Ohio']['Red'])
print

print('直接用MultiIndex创建层次索引结构')
print(MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Gree', 'Red', 'Green']],
                             names = ['state', 'color']))
