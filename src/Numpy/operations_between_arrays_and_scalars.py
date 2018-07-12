# -*- coding: utf-8 -*-

import numpy as np

# 数组乘法／减法，对应元素相乘／相减。
arr = np.array([[1.0, 2.0, 3.0], [4., 5., 6.]])
print(arr * arr) # 每个对应位置的元素相乘，注意这个要跟矩阵的乘法区别开。上次就理解错了
print(arr - arr)
print('\n')

# 标量操作作用在数组的每个元素上
arr = np.array([[1.0, 2.0, 3.0], [4., 5., 6.]])
print(1 / arr)
print(arr ** 0.5)  # 开根号 （每个元素都开根号）