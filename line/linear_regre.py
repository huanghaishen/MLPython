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
def linear_gradient():
    path = 'linear_data.csv'
    data = pd.read_csv(path, index_col=0)  # TV、Radio、Newspaper、Sales
    # 载入X变量 Y标签
    feature_cols = ['TV', 'radio', 'newspaper']
    x = data[feature_cols]
    y = data['sales']

    w=[]
    b=[]

    rate=0.01
    last_total_error=-1.0

    # 将数据进行归一化
    minMax = MinMaxScaler()
    x = minMax.fit_transform(x)



    #函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    len = np.shape(x_test)[0]
    num = np.shape(x_test)[1]
    w.append(np.random.uniform(-0.2, 0.2, num))
    b.append(np.random.uniform(-0.2, 0.2, 1))

    w=np.matrix(w)


    lent = np.shape(y_train)[0]
    y_train=np.dot(y_train,1)
    y_train=np.reshape(y_train,(lent,1))

    lentest = np.shape(x_test)[0]
    #梯度下降，梯度乘以学习率
    epoch=1
    while True:
        # 每次只用一个样本训练，随机梯度下降
        for m in range(lent):
            x_samble=x_train[m]
            y_samble=y_train[m]
            fx = np.dot(x_samble, w.T)+b
            # 计算误差
            fx=np.array(fx)
            delta_error = (fx -y_samble )

            # 求梯度
            grad_w = x_samble*delta_error
            grad_b = delta_error

            #更新权重和偏置项
            w=w-np.dot(rate,grad_w)
            b=b-np.dot(rate,grad_b)
        #检测总能的误差是否在下降，如果不下降，或者反而上升，则退出训练
        y_hat=np.dot(x_test,w.T)+b
        y_test_array=np.array(y_test)
        y_test_array=np.reshape(y_test_array,(lentest,1))

        #统计总误差
        total_error=y_hat-y_test_array
        total_error=np.dot(total_error.T,total_error).sum()

        print "epoch:"+str(epoch)+"    last_total_error:" + str(last_total_error) + " total_error:" + str(total_error)
        epoch=epoch+1
        if(last_total_error<=total_error and last_total_error>0 ):
            # 总误差不变，或者变大了，退出训练
            print "training finished"
            break
        else:
            last_total_error=total_error




    y_hat = np.dot(x_test, w.T)+b
    y_hat=np.array(y_hat)
    # plot y_label/y_predict

    t = np.arange(lentest)
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test(label)')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #linear_fun()
    #linear_matrix()
    linear_gradient()



