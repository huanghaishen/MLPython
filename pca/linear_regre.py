#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 线性回归
#换种说法,就是需要找到一个能够很好地将某些特征映射到其他特征的函数。依赖特征称为因
#变量,独立特征称为自变量
# 采用两种方法求解
# 第一种 用矩阵求逆求解W  linear_matrix
# 第二种 用梯度下降求解W
# 第三种 用坐标下降求解W
# 第四种 用牛顿迭代求解W
# 第五种 用拟牛顿法求解W


#小试牛刀，用线性回归预测

def linear_fun():
    path = 'linear_data.csv'
    data = pd.read_csv(path,index_col=0)    # TV、Radio、Newspaper、Sales

    print(data.head())
    feature_cols = ['TV', 'radio', 'newspaper']
    x = data[feature_cols]
    y = data['sales']



    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    linreg = LinearRegression()#获取线性回归的模型
    linreg.fit(x_train, y_train)#将数据填入模型中
    y_hat = linreg.predict(np.array(x_test))#将x_test转换成数组来做预测
    print(list(zip(feature_cols,linreg.coef_)))
    print('''
    对于给定了Radio和Newspaper的广告投入，如果在TV广告上每多投入1个单位，对应销量将增加0.0466个单位;
    更明确一点，假如其它两个媒体投入固定，在TV广告上每增加1000美元（因为单位是1000美元），销量将增加46.6（因为单位是1000）
    ''')
    #accuracy
    score = linreg.score(x_test,y_test)
    print("Accuracy :{:.2%}".format(score))
    #plot y_label/y_predict
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test(label)')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend()
    plt.show()

    print y_hat


# 利用矩阵求逆算法来求参
def linear_matrix():
    path = 'linear_data.csv'
    data = pd.read_csv(path, index_col=0)  # TV、Radio、Newspaper、Sales
    # 载入X变量 Y标签
    feature_cols = ['TV', 'radio', 'newspaper']
    x = data[feature_cols]
    y = data['sales']



    # 将数据进行归一化
    minMax = MinMaxScaler()
    x = minMax.fit_transform(x)

    # x添加bias项
    len = np.shape(x)[0]
    bx = np.ones(shape=[len, 1], dtype=int)
    # x = np.append(x,bx,axis=1)
    x = np.hstack((x, bx))


    #函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    #参数计算公式 θ=（X^TX)^-1X^TY
    #x_train=np.matrix(x_train)
    #y_train = np.matrix(y_train)
    xtx=np.dot(x_train.T,x_train)
    xtx_1=np.linalg.inv(xtx)

    xy=np.dot(x_train.T,y_train)
    W=np.dot(xtx_1,xy)

    y_hat = np.dot(x_test,W.T)


    # plot y_label/y_predict
    len=np.shape(x_test)[0]
    t = np.arange(len)
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test(label)')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend()
    plt.show()

    print y_hat



# 用梯度下降求解W
def linear_matrix():
    path = 'linear_data.csv'
    data = pd.read_csv(path, index_col=0)  # TV、Radio、Newspaper、Sales
    # 载入X变量 Y标签
    feature_cols = ['TV', 'radio', 'newspaper']
    x = data[feature_cols]
    y = data['sales']



    # 将数据进行归一化
    minMax = MinMaxScaler()
    x = minMax.fit_transform(x)

    # x添加bias项
    len = np.shape(x)[0]
    bx = np.ones(shape=[len, 1], dtype=int)
    # x = np.append(x,bx,axis=1)
    x = np.hstack((x, bx))


    #函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    #参数计算公式 θ=（X^TX)^-1X^TY
    #x_train=np.matrix(x_train)
    #y_train = np.matrix(y_train)
    xtx=np.dot(x_train.T,x_train)
    xtx_1=np.linalg.inv(xtx)

    xy=np.dot(x_train.T,y_train)
    W=np.dot(xtx_1,xy)

    y_hat = np.dot(x_test,W.T)


    # plot y_label/y_predict
    len=np.shape(x_test)[0]
    t = np.arange(len)
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test(label)')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend()
    plt.show()

    print y_hat



if __name__ == '__main__':
    linear_fun()
    #linear_matrix()

