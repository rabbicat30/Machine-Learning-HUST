
import os
import time
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score,learning_curve
import matplotlib.pyplot as plt
import numpy as np
import  pickle

# SVM Classifier,使用模型调参确定SVC的最佳参数
def svm_with_cross_validation(train_x, train_y):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    
    model = SVC(gamma=1)
    #打印学习曲线,仿照scikit的学习曲线所写

    plt.figure()
    plt.title('Learing Curve SVC rbf gamma:1, c:1')
    plt.xlabel('Training data')
    plt.ylabel('Score')

    train_size, train_score, test_score = learning_curve(model,train_x,train_y,cv=5,n_jobs=4,train_sizes=np.linspace(.1, 1.0, 5))
        
    train_score_mean = np.mean(train_score, axis=1)
    train_score_std =np.std(train_score,axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    test_score_std = np.std(test_score, axis=1)

    plt.grid()

    plt.fill_between(train_size, train_score_mean - train_score_std,
                     train_score_mean + train_score_std, alpha=0.1,color='r')
    plt.fill_between(train_size, test_score_mean - test_score_std,
                     test_score_mean + test_score_std, alpha=0.1,color='g')
    plt.plot(train_size, train_score_mean, 'o-', color="r",label="Training data")
    plt.plot(train_size, test_score_mean, 'o-', color="g",label="Cross validation score")
        
    plt.legend(loc="best")
    plt.show()

    model.fit(train_x, train_y)
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