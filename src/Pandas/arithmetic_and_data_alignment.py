# -*- coding: utf-8 -*- 

import numpy as np
from pandas import Series, DataFrame

print('加法')  # series的假发：只有索引的是一样的才会相加，索引不一样的会被设置成空值
s1 = Series([7.3, -2.5, 3.4, 1.5], index = ['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index = ['a', 'c', 'e', 'f', 'g'])
print(s1)
print(s2)
print(s1 + s2)  # 在对DataFrame做加法的时候，一行和列不匹配就会被设置成空值，也就是说2个DataFrame相加的时候行和列是有可能发生改变
print

print('DataFrame加法，索引和列都必须匹配。')
df1 = DataFrame(np.arange(9.).reshape((3, 3)),
                columns = list('bcd'),
                index = ['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12).reshape((4, 3)),
                columns = list('bde'),
                index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
print(df1)
print(df2)
print(df1 + df2)
print

print('数据填充')
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns = list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns = list('abcde'))
print(df1)
print(df2)
print(df1.add(df2, fill_value = 0))
print(df1.reindex(columns = df2.columns, fill_value = 0))
print

print('DataFrame与Series之间的操作')
arr = np.arange(12.).reshape((3, 4))
print(arr)
print(arr[0]) 
print(arr - arr[0])  # 每一行都减去arr[0]
frame = DataFrame(np.arange(12).reshape((4, 3)),
                  columns = list('bde'),
                  index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0] # 去除第一行
print(frame)
print(series)
print(frame - series) # 每一行减去series
series2 = Series(range(3), index = list('bef'))  # 任意一个值和缺失值相加减就认为他是一个缺失值NaN
print(frame + series2) 
series3 = frame['d']
print(frame.sub(series3, axis = 0))  # 按列减 frame减去series3,加减法的时候，默认是按行进行加减法，若指定axis=0,则变成按列进行加减法
