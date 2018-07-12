# -*- coding: utf-8 -*-
import numpy as np

print('Fancy Indexing: 使用整数数组作为索引')
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print(arr)
print(arr[[4, 3, 0, 6]]) # 打印arr[4]、arr[3]、arr[0]和arr[6]。
print(arr[[-3, -5, -7]]) # 打印arr[3]、arr[5]和arr[-7]行,python中-1下标代表倒数第一个元素
print('------------- arr -------------')
arr = np.arange(32).reshape((8, 4))  # 通过reshape变换成二维数组
print(arr)
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]]) # 打印arr[1, 0]、arr[5, 3]，arr[7, 1]和arr[2, 2]，二维的话，元素会一个一个的匹配

print('------------- test1 -------------')
print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]])  # 1572行的0312列，:表示1，5，7，2这些行都要打印对应0, 3, 1, 2的列，即第1行的0，3，1，2列...
print('------------- test2 -------------')
print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]) # 可读性更好的写法,跟上面的意思是一样