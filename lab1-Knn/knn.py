#coding:utf-8

import numpy as np
import os
import gzip
from six.moves import urllib
import operator
from datetime import datetime

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#处理数据
#newInput: vector to compare to existing dataset(1XM)
#dataSet: size m data set of known vectors (NXM)
#labels: data set labels(1XN vector)
#k: number of neighbors to use for comparision

def KnnClassify(newInput, dataSet, labels, k):
    #step1: 计算一直类别数据集当中的点与当前点之间的欧式距离
    numSamples = dataSet.shape[0]   #shape[0] stands for the num of row  N
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1,init_shape)
    diff = np.tile(newInput,(numSamples,1))-dataSet     #tile纵向拓展newInput向量使之变为NXM矩阵，再减去dataSet得到dataSet中的每一个点到newInput的每一个点的距离
    squareDiff=diff**2
    squareDist = np.sum(squareDiff,axis=1)  #得到的值按照行相加
    distance = squareDist**0.5

    #step2 ：按照距离递增的次序排序
    #argsort() 是将参数按照从小到大的顺序排列，提取其对应的索引输出到sortedDistIndices
    
    sortedDistIndices = np.argsort(distance)        
    classCount = {}     #define a empty dictionary(can be append element)
    for i in range(k):
        #step 3 :选出与当前点距离最小的k个点
        voteLabel = labels[sortedDistIndices[i]]    #得到这最小k个的类别：标签

        #step 4: 计算出这k个点所在类别出现的频率
        classCount[voteLabel] = classCount.get(voteLabel,0)+1   

    #step 5: 返回前k个点出现频率最高的类别最为当前点的预测分类
    maxCout=0
    maxIndex=0
    for key, value in classCount.items():
         if value >maxCout:
             maxCout = value 
             maxIndex = key
    #step2，3，4，5后来看网上其实可以简单的写成votes = Counter(sortedDistIndices[np.argpartition(distance,k)[:k]])
    #在返回votes.most_common(1)[0][0]   
    #其中votes可以视为一个dict，most_common（1）得到排在第一位的一个列表，取列表内的第一个元组，在取元组的第一个元素得到的就是类别
    #Counter需要加上from collection import Counter实现自动计算不同类别出现的频率，比如说Counter({1:7}),类别为1出现7次
    return maxIndex


    #下载mnist数据集
def maybe_download (filename, path, source_url):
    if not os.path.exists(path):
        os.makedirs(path)   #如果文件不存在创建文件
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
    return filepath
    
    #按32位读取，主要为读校验码、图片数量、尺寸准备的
    #仿照tensorflow的mnist.py写的
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4),dtype=dt)[0]

    #抽取图片，并按照需求可将图片中的灰度二值化后的数据村委矩阵或者张量
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s'%(magic,input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print(magic, num_images, rows, cols)
        buf = zipf.read(rows*cols*num_images)
        data = np.frombuffer(buf,dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data
    #抽取标签
    #仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels
    
maybe_download('train_images','data/mnist',SOURCE_URL+TRAIN_IMAGES)
maybe_download('train_labels','data/mnist',SOURCE_URL+TRAIN_LABELS)
maybe_download('test_images','data/mnist',SOURCE_URL+TEST_IMAGES)
maybe_download('test_labels','data/mnist',SOURCE_URL+TEST_LABELS)

    #主函数，读取图片再用于测试手写数字
def testHandWritingClass():
    #step 1:load data
    print('load data')
    train_x=extract_images('D:/VisualStdioCode_File/Python/machine learning/data/mnist/train_images',True,True)
    train_y=extract_labels('D:/VisualStdioCode_File/Python/machine learning/data/mnist/train_labels')
    test_x=extract_images('D:/VisualStdioCode_File/Python/machine learning/data/mnist/test_images',True,True)
    test_y=extract_labels('D:/VisualStdioCode_File/Python/machine learning/data/mnist/test_labels')

    #step2:trainning
    #print('step 2: trainning')
    #step 3:testing
    #print('step 3 :testing')
    a = datetime.now()
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = int(numTestSamples/10)
    for i in range (test_num):
        predict = KnnClassify(test_x[i],train_x,train_y,3)  #k=3
        if predict == test_y[i]:
            matchCount += 1
        #if i%100 == 0:
        #    print('完成%d张图片'%(i))
    accuracy = float(matchCount) / test_num
    b = datetime.now()
    print('the time we use is %ds'%((b-a).seconds))

    #step 4： show the result
    #print('step 4: show the result')
    print('The accuracy is %.2f%%'%(accuracy*100))

if __name__ == '__main__':
    testHandWritingClass()