import re
import numpy as np
import random 

"""
函数说明：实现将一封邮件进行分词，即划分为一个个单词的形式,正则表达式只允许返回单词长度>2的单词
参数：邮件名
返回参数：一个分词的列表
"""
def textParse(bigString):
    listofTokens=re.split(r'\W',bigString)
    return [tok.lower() for tok in listofTokens if len(tok)>2]

"""
函数说明：将所有的邮件进行分词后生成一个dataSet，然后再生成一个词汇表，该词汇表为集合
参数：dataSet--所有邮件被分词后形成的总列表
返回：词汇表列表
"""
def creatVocablist(dataSet):
    vocabSet = set([])
    for word in dataSet:
        vocabSet = vocabSet|set(word)   #两个集合取交集
    return list(vocabSet)   #将其转化为一个列表

"""
函数说明：词汇表生成后将每一封邮件生成一个词向量，词向量的元素的值位每个单词出现的次数
参数：vocablist--词汇表，inputSet输入的邮件
返回参数：词向量--维度和词汇表相同
"""
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

"""
函数说明：训练算法，计算在垃圾邮件下每一个特征的频率和非垃圾邮件下每一个特征的频率
          这里的特征指的是单词
参数：trainMat--训练样本的词向量矩阵，其每一行为一个邮件的词向量，trainCategory为与trainMat对应的类别矩阵
返回参数：属于正常邮件类的条件概率数组和垃圾邮件类的条件概率数组，文档属于垃圾邮件的概率
"""
def train (trainMat, trainCategory):
    numTrain = len(trainMat)#训练的邮件的数目
    numWords = len(trainMat[0])#每份邮件的词条数
    pAbusive = sum(trainCategory) / float (numTrain)#文档属于垃圾邮件的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)#创建ones数组，词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0   #分母初始化为2，拉普拉斯平滑
    for i in range(numTrain):
        if trainCategory[i]==1:
            p1Num += trainMat[i]#类1中每个单词的个数
            p1Denom += sum(trainMat[i])#类1的单词总数
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vec = np.log(p1Num/p1Denom)
    p0Vec = np.log(p0Num/p0Denom)
    return p0Vec,p1Vec,pAbusive

"""
函数说明：处理数据验证过程，将邮件镀金doclist列表中，然后生成一个词汇表包含所有的单词
          然后使用交叉验证

"""
def spamTest ():
    docList = []
    fullText = []
    classList = []
    for i in range(1,26):#遍历25个txt文件
        wordList = textParse(open('D:\Save_for_English\email\spam\%d.txt' %i).read())#读取每一个垃圾邮件，并将字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)#标记垃圾邮件
        wordList = textParse(open('D:\Save_for_English\email\ham\%d.txt' %i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = creatVocablist(docList)#创建总的词汇表
    trainingSet = list(range(50))
    testSet = []#创建存储训练集的索引的列表和测试集的索引值的列表
    for i in range(10):#从50个邮件中，随机抽取40个作为训练集，10个作为测试集
        randIndex = int(random.uniform(0,len(trainingSet)))#随机选取索引值
        testSet.append(trainingSet[randIndex])#添加测试集的索引值
        del (trainingSet[randIndex])#在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []#创建训练集矩阵和训练集类别标签向量
    for docIndex in trainingSet:#遍历训练集
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))#将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])#将类别添加到训练集类别标签系向量中
    p0,p1,pSpam = train(np.array(trainMat),np.array(trainClasses))#训练朴素贝叶斯模型
    errCount = 0
    for docIndex in testSet:#遍历测试集
        wordVec = bagOfWords2Vec(vocabList,docList[docIndex])#测试集的词集模型
        if classfy(np.array(wordVec),p0,p1,pSpam) != classList[docIndex]:#分类错误
            errCount +=1
            print("classfication error"),docList[docIndex]
    print('the error rate is %.2f%%' % (float(errCount)/len(testSet)*100))

"""
函数说明：朴素贝叶斯分类器分类函数
参数：
vec2Classfify--待分类的词条数组
p0Vec--正常邮件的条件概率数组
p1Vec--垃圾邮件的条件概率数组
pClass1--文档为垃圾邮件的概率
返回参数：邮件分类
"""
def classfy(vec2Classfify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classfify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classfify*p0Vec)+np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    spamTest()