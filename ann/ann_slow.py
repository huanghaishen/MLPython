#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import random
import time
from numpy import *
import numpy as np
from dataset import *







class Network( object):
    def __init__(self ,layers):
        # 神经网络的层列表
        self.rate=0.5
        self.layer_num=0
        self.layer_left=0
        self.layers = []
        self.links_weight=[]
        self.links_b=[]
        self.layer_num = len(layers)

        self.output = map(lambda x: np.zeros((x, 1)), layers[:])
        self.delta = map(lambda x: np.zeros((x, 1)), layers[:])
        self.gradient = map(lambda x: np.zeros((x, 1)), layers[1:])


        for i in range(1,self.layer_num):

            self.links_weight.append(np.random.uniform(-0.2, 0.2, (layers[i], layers[i-1])))
            self.links_b.append(np.random.uniform(-0.2, 0.2, layers[i]))

        #self.links_b = np.array([np.array([1.0, 1.0]), np.array([1.0, 1.0])])



        #self.links_weight = np.array([np.array([[1.0, -1.0],
        #                        [2.0, -2.0]]),
        #              np.array([[1.0, -1.0],
        #                        [2.0, -2.0]])])

    def sigmoid_(self,inx):
        if inx >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
            return 1.0 / (1 + exp(-inx))
        else:
            return exp(inx) / (1 + exp(inx))

    def sigmoid(self,inX):

        #inX_=np.dot(inX,-1)
        return np.array(map(lambda x: self.sigmoid_(x),inX))



    def forward(self,sample):


        #print sample
        out=sample
        self.output[0]=out

        for i in range(1,self.layer_num):

            #权重乘输入
            weight=self.links_weight[i-1]
            out=np.dot(weight, out)
            #加偏置项
            b=self.links_b[i-1]
            #out = map(lambda x, y: x + y, out, b)
            net = out+b
            out = self.sigmoid(net)
            self.output[i]=out

        return out

    def backward(self,label):
        # 计算delta
        output = self.output[-1]

        dd = label - output
        dd2= 1.0 - output
        dd_= -output
        delta = -output * (1.0 - output) * (label - output)

        self.delta[-1] = delta

        # print self.delta[-1]

        # 往前传播 误差
        for t in range(1, self.layer_num):
            ii = -t
            # 汇总下游delta
            if ii > 1 - self.layer_num:
                weight = self.links_weight[ii]
                d = self.delta[ii]
                output_ii=np.array(self.output[ii-1])

                output1_=1.0-output_ii

                downdelta = np.dot(weight.T, d)
                out_h1=output_ii*(1.0-output_ii)
                downdelta_= out_h1*downdelta
                self.delta[ii - 1] = downdelta_

            # 计算 梯度

            #d_c = np.mat(self.delta[ii]).T
            #o_c = np.mat(self.output[ii - 1])
            #gra = np.array(np.dot(d_c,o_c))

            d_c = np.mat(self.delta[ii]).T
            o_c = np.mat(self.output[ii - 1])
            gra = np.array(np.dot(d_c,o_c))




            self.gradient[ii] = gra
            # 更新输出层权重
            rate_gra = self.rate * self.gradient[ii]
            self.links_weight[ii] = self.links_weight[ii] - rate_gra
            rate_b = self.rate * self.delta[ii]
            self.links_b[ii] = self.links_b[ii] - rate_b

            d=1



    def train(self, labels, data_set, iteration):
        '''
        训练神经网络
        labels: 数组，训练样本标签。每个元素是一个样本的标签。
        data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        '''
        for i in range(iteration):

            for d in range(len(data_set)):
                if d == 3000 :
                    ff=1
                self.forward(data_set[d])
                self.backward(labels[d])



    def print_network(self):
        for i in range(self.layer_num):
            layer=self.layers[i]
            for ii in range(len(layer.nodes)):
                print(str(layer.nodes[ii].layer) + '>>' + str(layer.nodes[ii].no))



def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('data/train-images-idx3-ubyte', 60000)
    label_loader = LabelLoader('data/train-labels-idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()
def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('data/t10k-images-idx3-ubyte', 10000)
    label_loader = LabelLoader('data/t10k-labels-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.forward(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def train_and_evaluate():
    print("start train....")
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 30, 10])
    #network = Network([2, 2, 2])
    network.rate = 0.03

    #network.train(np.array([np.array([0,1])]), np.array([np.array([1,2])]), 2)



    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 10)
        print('%s epoch %d finished' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()
