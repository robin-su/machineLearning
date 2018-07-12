# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

'''
    
'''
if __name__ == '__main__':
    # pandas 读入
    data = pd.read_csv('D:\machineLearning\src\zhouboML\Regression\data\8.Advertising.csv')
    x = data[['TV','Radio','Newspaper']]
    y = data['Sales']

    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1) # 划分训练集和测试集,默认train_size=0.75
    model = Lasso() # 使用L1-norm
    # model = Ridge() # 使用L2-norm

    # Lasso是需要alpha_can，因此我们先列出alpha_can（超参数）
    alpha_can = np.logspace(-3,2,10)  # 从10^-3到10^2抽出10个数，使得这10个数是等比数列
    '''
        GridSearchCV表示做交叉验证，看看哪个alpha_can能使得模型的效果最好，
        cv=5表示5折的交叉验证（将训练数据劈成5份，每4份做训练，一份做验证），lasso_model就是选出最好的那个模型
    '''
    lasso_model = GridSearchCV(model,param_grid={'alpha':alpha_can},cv=5)
    lasso_model.fit(x_train,y_train)
    print("验证参数：\n",lasso_model.best_params_) #{'alpha': 2.1544346900318843}

    y_hat = lasso_model.predict(np.array(x_test))
    # print(lasso_model.score(x_test,y_test)) # 打印分数，其实就是决定系数
    mse = np.average((y_hat - np.array(y_test)) ** 2) # Mean Squared Error
    rmse = np.sqrt(mse) # Root Mean Squared Error

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

