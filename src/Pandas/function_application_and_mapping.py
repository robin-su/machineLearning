# -*- coding: utf-8 -*- 

import numpy as np
from pandas import Series, DataFrame

print('函数')
frame = DataFrame(np.random.randn(4, 3),
                  columns = list('bde'),
                  index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
print(frame)
print(np.abs(frame)) # 取绝对值
print

print('lambda以及应用')
f = lambda x: x.max() - x.min()
print(frame.apply(f)) # 按列操作。则具体操作为每一列的最大值减去最小值
print(frame.apply(f, axis = 1)) # 表示每一行的最大值减去最小值,注意正常我们axis=0表示行，但是我们这里可以这样理解，他时间行固定，计算列
def f(x):
    return Series([x.min(), x.max()], index = ['min', 'max']) # 以min和max为索引
print(frame.apply(f))
print

print('applymap和map')
_format = lambda x: '%.2f' % x #将x按照%.2f的浮点百分号去打印
print(frame.applymap(_format)) # applymap对于每个元素都按照上面定义的函数去做
print(frame['e'].map(_format)) # 对于‘e’列中的每个元素都按照_format函数去朝族
# 对于 frame.applymap就是作用到没一个元素
# frame['e']实际上是一个series数据类型，也就是对于使用map就可以映射每个元素