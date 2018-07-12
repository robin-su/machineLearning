# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

'''
    简易版本的线性回归
'''
if __name__ == "__main__":
    path = "D:\machineLearning\src\zhouboML\Regression\data\8.Advertising.csv"

    # 使用pandas读入
    data = pd.read_csv(path)
    x = data[['TV','Radio','Newspaper']] # TV,Radio,Newspaper
    y = data['Sales']
    # print(x)
    # print(y)

    ## 绘制1
    # plt.plot(data['TV'],y,'ro',label='TV')
    # plt.plot(data['Radio'],y,'g^',label='Radio')
    # plt.plot(data['Newspaper'],y,'mv',label='Newspaper')
    # plt.grid()
    # plt.show()

    ## 绘制2
    # plt.figure(figsize=(9,20))
    # plt.subplot(311)
    # plt.plot(data['TV'],y,'ro')
    # plt.title('TV')
    # plt.grid()
    #
    # plt.subplot(312)
    # plt.plot(data['Radio'], y, 'g^')
    # plt.title('Radio')
    # plt.grid()
    #
    # plt.subplot(313)
    # plt.plot(data['Newspaper'],y,'b*')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.show()

    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
    # print(x_train)
    # print(y_train)
    linreg = LinearRegression()
    model = linreg.fit(x_train,y_train)
    print(model)
    print(linreg.coef_) # 打印系数 [ 0.04656457  0.17915812  0.00345046]
    print(linreg.intercept_) # 截距  2.87696662232

    y_hat = linreg.predict(np.array(x_test)) # 测试集合
    mse = np.average((y_hat - np.average(y_test)) ** 2) # 均方误差
    rmse = np.sqrt(mse)
    print("mse: " + str(mse) + " rmse:" + str(rmse))

    # print(np.arange(len(x_test)))
    t = np.arange(len(x_test))
    plt.plot(t,y_test,'r-',linewidth=2,label='Test')
    plt.plot(t,y_hat,'g-',linewidth=2,label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
