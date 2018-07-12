# -*- coding: utf-8 -*- 

import numpy as np
from pandas import DataFrame, Series

print('重新指定索引及顺序')
obj = Series([4.5, 7.2, -5.3, 3.6], index = ['d', 'b', 'a', 'c'])
print(obj)
obj2 = obj.reindex(['a', 'b', 'd', 'c', 'e'])
print(obj2)
print(obj.reindex(['a', 'b', 'd', 'c', 'e'], fill_value = 0))  # 指定不存在元素的默认值
print

print('重新指定索引并指定填元素充方法')
obj3 = Series(['blue', 'purple', 'yellow'], index = [0, 2, 4])
print(obj3)
print(obj3.reindex(range(6), method = 'ffill')) # ffill 就是若这一行的值是空的值,那么我就用前面一行的值去填充
print

print('对DataFrame重新指定索引')
frame = DataFrame(np.arange(9).reshape(3, 3),
                  index = ['a', 'c', 'd'],
                  columns = ['Ohio', 'Texas', 'California'])
print(frame)
frame2 = frame.reindex(['a', 'b', 'c', 'd'])  # 重新建索引的时候，因为'b'没有索引，会被用NaN代替
print(frame2)
print

print('重新指定column')
states = ['Texas', 'Utah', 'California']
print(frame.reindex(columns = states))
print

print('对DataFrame重新指定索引并指定填元素充方法')
print(frame.reindex(index = ['a', 'b', 'c', 'd'],  # 'b'这一行的数据为空，本来是用NaN去填充，但是这里指定了为ffill，是用一行去填充的
                    method = 'ffill',
                    columns = states))
print(frame.ix[['a', 'b', 'd', 'c'], states])  #增加了一行b，所以b的值为NaN,state中Ohio已经去掉了，这里会消失，新增的Utah列会变成NaN
