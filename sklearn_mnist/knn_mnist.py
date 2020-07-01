from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from skimage import exposure
import matplotlib.pyplot as plt 
import numpy as np 
import imutils
import cv2
import time
from sklearn.preprocessing import StandardScaler

#加载mnist数据集
data,target = datasets.load_digits(return_X_y=True)

#训练数据和测试数据分隔，其中75%为训练数据，25%为测试数据
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), target, test_size=0.25, random_state=42)
"""
ss = StandardScaler()
trainData = ss.fit_transform(trainData)
testData = ss.transform(testData)
"""
#take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData,trainLabels,test_size=0.1,random_state=84)

#checking sizes of each data split
print("trainging data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing datapoins: {}".format(len(testLabels)))


#初始化k的值并计算每一个k值对应的准确率
kVals = range(1,30,2)
accuracies = []

#循环
for k in range(1,30,2):
    #train the classfier with the current value of 'k'
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData,trainLabels)

    #评估模型并打印准确度
    score = model.score(valData,valLabels)
    print("k=%d, accuracy=%.2f%%" % (k,score*100))
    accuracies.append(score)

#largest accuracy
#np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i]*100))


#现在我已经知道最好的k值，重新训练分类器
a = time.time()
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData,trainLabels)

#测试验证集
predictions = model.predict(testData)
print('the time we use: %fs' %(time.time()-a))

#Evaluate performance of model for each of the digits
print("EVLUATION ON TESTING DATA")
print(classification_report(testLabels,predictions))

#打印混淆矩阵
labels1 = list(set(predictions))
conf_mat1 = confusion_matrix(testLabels,predictions,labels=labels1)
print(conf_mat1)

#check predictions against images
#loop over a few random digits 
image = testData
j = 0
for i in np.random.randint(0,high=len(testLabels),size=(24,)):
    #np.random.randint(low, high=None,size=None,dyte='1')
    prediction = model.predict(image)[i]
    image0 = image[i].reshape((8,8)).astype("uint8")
    image0 = exposure.rescale_intensity(image0, out_range=(0,255))
    plt.subplot(4,6,j+1)
    plt.title(str(prediction))
    plt.imshow(image0,cmap='gray')
    plt.axis('off')

    #convert the image for a 64-dim array to an 8x8 image compatible with OpenCV,
    #the resize it to 32 x 32 pixels for better visualization

    #image0 = imutils.resize(image[0],,width=32,inter=cv2.INTER_CUBIC)

    j= j+1

#show prediction
plt.show()

