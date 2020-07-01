from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import numpy as np 
import imutils
import time 
import  pickle

#读取mnist数据，文件中包括50000个训练数据和10000个测试数据，基于sklearn的数据太少了，感觉对比不是很明显
def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    Myunpickle = pickle._Unpickler(file = f, fix_imports=True, encoding="bytes", errors="strict")
    train,val,test = Myunpickle.load()
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

#初始化k的值并计算每一个k值对应的准确率
kVals = range(1,30,2)
accuracies = []

#载入数据，这个mnist文件是从网上找的，训练数据仅有50000，比真正的mnist训练集少了10000，但是找官网的话好麻烦就直接拿这个了
data_file = "D:\VisualStdioCode_File\Python\sklearn_mnist\mnist.pkl.gz"#不知道为什么即便放在同一个文件夹下面也不可以直接写mnist.pk.gz
traindata, trainlabels, testdata, testlabels = read_data(data_file) 
print("training data : %d, testing data : %d, dimesion : %d" %(traindata.shape[0],testdata.shape[0],traindata.shape[1]))
#循环
import matplotlib.pyplot as plt 
for k in range(1,30,2):
    #从训练数据中得到模型
    start_time = time.time()
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(traindata,trainlabels)
    print("for k=%d, the time used for training is %fs" %(k,time.time()-start_time))

    #用得到的模型测试测试集
    begin_time = time.time()
    y_predict = model.predict(testdata)
    print('          the time we use for testing is : %fs' %(time.time()-begin_time))
    print("          accuracy=%.2f" % (metrics.accuracy_score(testlabels, y_predict)))

    #打印混淆矩阵
    labels1 = list(set(y_predict))
    conf_mat1 = confusion_matrix(testlabels,y_predict,labels=labels1)
    print(conf_mat1)

    #Evaluate performance of model for each of the digits
    print("EVLUATION ON TESTING DATA")
    print(classification_report(testlabels,y_predict))

