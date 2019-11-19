#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from PIL import Image
import numpy as np
def loadImage(path):
    img = Image.open(path)
    # 将图像转换成灰度图
    img = img.convert("L")
    # 图像的大小在size中是（宽，高）
    # 所以width取size的第一个值，height取第二个
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    # 为了避免溢出，这里对数据进行一个缩放，缩小100倍
    #data = np.array(data).reshape(height,width)/100
    # 查看原图的话，需要还原数据
    #new_im = Image.fromarray(np.uint8(data*100))
    #img.show()
    return data,width,height

def showImage(path):
    img = Image.open(path)
    # 将图像转换成灰度图
    img = img.convert("L")

    img.show()


#  白化数据 ，每行元素 减去均值
def center(data,width,height):
    data_1=np.ones((height,width) )


    
    data_avg=np.mean(data,axis=1)
    #print data
    data_white= data_1.transpose()*data_avg
    data_avg=data_white.transpose()

    data_white= data-data_avg

    return data_white,data_avg

#  计算协方差矩阵
#  协方差指的是样本的属性和属性的协方差，所以，协方差矩阵最后是一个方阵，高度和宽度是属性的数量
def cov(data,width):
    covdata=np.dot(data,data.T)
    covdata=covdata/(width-1)
    return covdata


# 计算特征值
def eig(data):
    a,b=np.linalg.eig(data)
    return a,b


# 获取前k个特征值对应的特征向量
def geteig_k(eig_value,eig_vector,k):



   # new_vector=un_eig[1]

    eigIndex = np.argsort(eig_value)
    eigVecIndex = eigIndex[:-(k + 1):-1]
    feature = eig_vector[:, eigVecIndex]
    #new_data = np.dot(normal_data, feature)



    gg=eig_value[0:k].sum()/eig_value[:].sum()
    print "前"+str(k)+"个特征值贡献率为："+str(gg)

    return feature

    #print neweig

def testarray():
    a=np.array([[11,12,13,14],[21,22,23,24],[31,32,33,34]])
    b=np.array([[45],[55],[65]])
    c=np.hstack((a,b))
    print c



def pca(imgpath):
    # width为样本数 m 430
    # height为属性数n 291
    data, oldwidth, oldheight = loadImage(imgpath)
    dim = 96

    width = oldwidth * oldheight / dim
    height = dim

    # 形状可以随便指定
    data_shape = np.array(data).reshape(height, width)

    wd, data_avg = center(data_shape, width, height)

    covdata = cov(wd, width)
    # covdata1=np.cov(wd)

    eig_value, eig_vector = eig(covdata)

    w = geteig_k(eig_value, eig_vector, 90)

    # 投影到新坐标
    new_y = np.dot(w.T, wd)

    # 重构图像

    new_x = np.dot(w, new_y)

    # 加上平均值
    image_data = new_x + data_avg

    data_shape = image_data.reshape(oldheight, oldwidth)
    img = Image.fromarray(data_shape)
    img.show()

def svd(imgpath,k):
    data, oldwidth, oldheight = loadImage(imgpath)

    data_shape = np.mat(np.array(data).reshape(oldheight,oldwidth))



    u, s, v = np.linalg.svd(data_shape, full_matrices=1, compute_uv=1)

    #s 是一个数组，需要对角化


    sk=np.diag(s[:k])


    vk=v[:k,:]
    svk=np.dot(sk,vk)

    uk=u[:,:k]
    newdata=np.dot(uk,svk)


    #计算贡献率
    rate=np.sum(sk)/np.sum(s)
    print "贡献率" + str(round((rate * 100), 2)) + "%"

    img = Image.fromarray(newdata)
    img.show()





if __name__ == '__main__':

    imgpath = "/Users/Shared/学习/code/image/lake.jpeg"

    # 采用pca方法来压缩图片
    #pca(imgpath)

    #采用svd方法来压缩图片

    svd(imgpath,200)
