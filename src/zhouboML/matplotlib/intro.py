# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# 绘制正态分布概率密度函数
def drawingGuass():
    '''这两行是为了让中文字体可以显示用的'''
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  #FangSong/黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False # 符号不要特殊处理

    mu = 0 # 均值是0
    sigma = 1 # 方差是1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50) # 抽出-3到+3中的50个数作为绘制点
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma) # 正太分布函数
    print(x.shape)
    print('x = \n',x)
    print(y.shape)
    print('y = \n',y)
    ''' r- 表示使用红色的线；
        go表示绿色的圆圈；
        linewidth表示线宽是2
        markersize表示圈的大小是8
    '''
    plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
    plt.grid(True) # 画出网格
    plt.title(u"Guass分布")
    plt.show()

# 绘制损失函数
def drawingLoss():
    x = np.array(np.linspace(start=-2,stop=3,num=1001,dtype=np.float))
    y_logit = np.log(1 + np.exp(-x)) / math.log(2)
    y_boost = np.exp(-x)
    y_01 = x < 0
    y_hinge = 1.0 - x
    y_hinge[y_hinge < 0] = 0
    # label 标签用于表示线条的表示方法
    plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
    plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
    plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
    plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2) # m--表示紫色的虚线
    plt.grid(True)
    plt.legend(loc='upper right')# 表示label标签要放置在图像的右上角
    plt.savefig('1.png')  # 保存图片
    plt.show()

# 绘制心型图
def drawingHeartPatter():
    x = np.linspace(-1.3,1.3,101)
    y = np.ones_like(x) # 依据给定数组(x)的形状和类型返回一个新的元素全部为1的数组。
    i = x > 0
    y[i] = np.power(x[i],x[i]) # power(x[i],x[i]) 表示x[i]的x[i]次方
    i = x < 0
    y[i] = np.power(-x[i],-x[i])
    plt.plot(x,y,'g-',label='x^x',linewidth=2)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

'''
    数值计算实例：
        对于某2分类问题，若构造了10个正确率都是0.6的分类器，采用少数服从多数的原则进行最终分类，
        则最终分类的正确率是多少？
        
        若构造100个分类器呢？
'''
