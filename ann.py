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


    # 预测输出
    def forward(self,sample):



        self.output[0]=sample

        for i in range(1,self.layer_num):
            #权重乘输入

            self.output[i] = self.sigmoid(np.dot(self.links_weight[i-1], self.output[i-1])+self.links_b[i-1])

        return self.output[self.layer_num-1]

    # 反向传播，更新权重
    def backward(self,label):
        # 计算delta

        self.delta[-1] = -self.output[-1] * (1.0 - self.output[-1]) * (label - self.output[-1])

        # print self.delta[-1]

        # 往前传播 误差，从后往前传的误差，经过激活函数求导后，变成新的delta，
        # 这个delta保存下来，有两个作用，
        # 一个用来更新本层的输入连接，delta乘以输入的函数（wx+b）对w的求导，其实就是x，
        # 一个用来继续往前对传递，delta乘以输入的函数（wx+b）对x的求导，其实就是权重w，
        for t in range(1, self.layer_num):
            ii = -t
            # 汇总下游delta
            if ii > 1 - self.layer_num:
                output_ii=np.array(self.output[ii-1])
                self.delta[ii - 1] = output_ii*(1.0-output_ii)*np.dot(self.links_weight[ii].T, self.delta[ii])

            # 计算 梯度

            self.gradient[ii] = np.array(np.dot(np.mat(self.delta[ii]).T,np.mat(self.output[ii - 1])))
            # 更新输出层权重

            self.links_weight[ii] = self.links_weight[ii] - self.rate * self.gradient[ii]

            self.links_b[ii] = self.links_b[ii] - self.rate * self.delta[ii]


    def train_one_sample(self,sample,label):
        self.forward(sample)
        self.backward(label)

    def train(self, labels, data_set, iteration):
        '''
        训练神经网络
        labels: 数组，训练样本标签。每个元素是一个样本的标签。
        data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        '''
        for i in range(iteration):
            map(lambda s,l:self.train_one_sample(s,l),data_set,labels)
        a=1





    def print_network(self):
        for i in range(self.layer_num):
            layer=self.layers[i]
            for ii in range(len(layer.nodes)):
                print(str(layer.nodes[ii].layer) + '>>' + str(layer.nodes[ii].no))



def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('data/train-images-idx3-ubyte', 6000)
    label_loader = LabelLoader('data/train-labels-idx1-ubyte', 6000)
    return image_loader.load(), label_loader.load()
def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('data/t10k-images-idx3-ubyte', 1000)
    label_loader = LabelLoader('data/t10k-labels-idx1-ubyte', 1000)
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
    network.rate = 0.01

    #network.train(np.array([np.array([0,1])]), np.array([np.array([1,2])]), 2)

    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 20)
        print('%s epoch %d finished' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, error_ratio))
            if error_ratio >= last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()
    
