#!usr/bin/env python
#-*- coding: utf-8 -*-
import os
import time
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import  pickle
 
# Multinomial Naive Bayes Classifier
def Multionmial_naive_bayes(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)#lidstone平滑，因为平滑项相当于人为加入噪声，数值越大的话准确度会越低
    model.fit(train_x, train_y)
    return model
 
 #Gaussian Navie Bayes Classifier
def Gaussian_naive_bayes(train_x, train_y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model

#Bernoulli Navie Bayes Classifier
def Bernoulli_navie_bayes(train_x, train_y):
    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model

 
# KNN Classifier
def knn(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=103)
    model.fit(train_x, train_y)
    return model
 
 
# Logistic Regression Classifier
def logistic_regression(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model
 
 
 
# SVM Classifier,使用模型调参确定SVC的最佳参数
def svm_with_cross_validation(train_x, train_y):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    best_score = 0
    cv_scores = []
    C = [1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4]
    Gamma = [1e-4,1e-3,1e-2,1e-1,1,10]
    for gamma in Gamma:
        model = SVC(gamma=gamma)
        print('gamma=%s'%gamma)
        scores = cross_val_score(model, train_x,train_y, cv=5,scoring='accuracy')#使用5折交叉验证
        print('cross validatin over')
        score = scores.mean()
        print('score : %f'%score)
        cv_scores.append(score)
        if score >best_score:
            best_score = score
            best_params = {'gamma':gamma}
    print('Best parameters :{}'.format(best_params))
    plt.plot(Gamma,cv_scores)
    plt.xlabel('gamma')
    plt.ylabel('Accuarcy')
    plt.show()
    model = SVC(C=best_params["gamma"])

    print('fit begin')
    model.fit(train_x, train_y)
    print('fit over')
    return model
 
 
def read_data(data_file):
    import gzip

    f = gzip.open(data_file, "rb")
    Myunpickle = pickle._Unpickler(file = f, fix_imports=True, encoding="bytes", errors="strict")
    train,val,test = Myunpickle.load()#train是一个由两个元素组成的元组，第一个元素是测试图片的额集合，为50000*784的矩阵，每一行表示一个数据，第二个元素是测试图片的标签
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y
    
if __name__ == '__main__':
    data_file = "D:\VisualStdioCode_File\Python\sklearn_mnist\mnist.pkl.gz"
    thresh = 0.5
    model_save_file = None
    model_save = {}
    """
    test_classifiers = ['Navie Bayes', 'KNN', 'Logistic Regression',  'SVM']
    classifiers = {'Navie Bayes':naive_bayes, 
                   'KNN':knn,
                   'Logistic Regression':logistic_regression,
                   'SVM':svm_with_gridsearchcv,
    }
    """
    test_classifiers = [ 'SVM']
    classifiers = {'SVM':svm_with_cross_validation,
    }
    print ('reading data...')
    train_x, train_y, test_x, test_y = read_data(data_file)
    is_binary_class = (len(np.unique(train_y)) == 2)
    print ("#training data: %d, #testing_data: %d, dimension: %d" % (train_x.shape[0], test_x.shape[0], train_x.shape[1]))
    for classifier in test_classifiers:
        print (" %s"  % classifier)

        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print ("the time used for training is %fs" % (time.time() - start_time))

        begin_time = time.time()
        predict = model.predict(test_x)
        print("the time used for predicting is %fs" %(time.time()-begin_time))

        accuracy = metrics.accuracy_score(test_y, predict)#直接使用自带的评估函数
        print ("accuracy: %.2f%%" % (100 * accuracy)) 

        #打印混淆矩阵
        labels1 = list(set(predict))
        conf_mat1 = confusion_matrix(test_y,predict,labels=labels1)
        print(conf_mat1)
        print(classification_report(test_y,predict))#打印精准度召回率等问题

        """
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print ("precision: %.2f%%, recall: %.2f%%" % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print ("accuracy: %.2f%%" % (100 * accuracy)) 
 
    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
    """