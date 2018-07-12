# -*- coding:utf-8 -*-
import operator
from functools import reduce

def c(n,k):
    '''
       operator.mul 表示乘积，这里计算的是从n-k+1到n的乘积
    '''
    return reduce(operator.mul, range(n-k+1, n+1)) / reduce(operator.mul, range(1, k+1))


def bagging(n, p):
    s = 0
    for i in range(n // 2 + 1, n + 1): # //表示Floor除法总是省略小数部分
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s

if __name__ == '__main__':
    for t in range(9,100,10):
        print(str(t)+" 次采样正确率为：" + str(bagging(t,0.6)))