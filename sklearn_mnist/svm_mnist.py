from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np 
from datetime import datetime

#加载mnist数据集
digits = datasets.load_digits()
print(digits)
#训练数据和测试数据分隔，其中75%为训练数据，25%为测试数据
(trainData, testData, trainLabels, testLabels) = train_test_split(digits["data"],digits["target"], test_size=0.25, random_state=42)


#checking sizes of each data split
print("trainging data points: {}".format(len(trainLabels)))
print("testing datapoins: {}".format(len(testLabels)))

#从仍需要对训练和测试的特征数据进行标准化
ss = StandardScaler()
trainData = ss.fit_transform(trainData)
testData = ss.transform(testData)

a = datetime.now()
#初始化先行假设的支持向量机分类器LinearSVC
svc = LinearSVC()

#进行模型训练
svc.fit(trainData,trainLabels)

#进行预测
y_predict = svc.predict(testData)

print('the time we use is %d' %((datetime.now()-a).seconds))
#打印混淆矩阵
labels1 = list(set(y_predict))
conf_mat1 = confusion_matrix(testLabels,y_predict,labels=labels1)
print("SVM")
print(conf_mat1)

#使用模型自带的评估函数进行准确性测评
print("The Accuarcy of Linear SVC is %.2f" %(svc.score(testData,testLabels)))

#使用classfiction_report模块对预测结果进行分析
print(classification_report(testLabels,y_predict))
"""
#保存模型
from sklearn.externals import joblib
joblib.dump(svc, 'D:/VisualStdioCode_File/Python/sklearn_mnist')
#载入保存的模型
svc2 = joblib.load('D:/VisualStdioCode_File/Python/sklearn_mnist')

#预测
y_pred = svc2.predict(testData)

print(y_pred)
"""
