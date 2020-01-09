#!/usr/bin/env python
# coding: utf-8

# # CSE 258, Fall 2019: Homework 2

# Student ID: A53308934 <br>
# Student Name: Deng Zhang

# ## Tasks - Diagnostics

# In[1]:


import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
from random import shuffle
import random


# In[2]:


f = open("./Data/hw2/5year.arff", 'r')


# In[3]:


while not '@data' in f.readline():
    pass


# In[4]:


dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# ### Question 1: 
# Code to read the data is available in the stub. Train a logistic regressor (e.g. sklearn.linear model.LogisticRegression) with regularization coefficient C = 1.0. Report the accuracy and Balanced Error Rate (BER) of your classifier.

# In[5]:


# use the last col as y, the reset as x
X = [values[:-1] for values in dataset]
y = [values[-1] for values in dataset]


# In[6]:


model = linear_model.LogisticRegression(C=1.0)


# In[7]:


model.fit(X, y)


# In[8]:


predictions = model.predict(X)

correct = predictions == y

print("Accuracy = " + str(sum(correct) / len(correct)))


# In[9]:


TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])


# In[10]:


TPR = TP / (TP + FN)
TNR = TN / (TN + FP)


# In[11]:


BER = 1 - 1/2 * (TPR + TNR)
print("Balanced error rate = " + str(BER))


# In[12]:


# Answer of Question 1:

# Accuracy = 0.9663477400197954
# Balanced error rate = 0.48580623782459387


# ### Question 3:

# In[77]:


random.shuffle(dataset)


# In[78]:


X = [values[:-1] for values in dataset]
y = [values[-1] for values in dataset]
N = len(X)
X_train = X[:N//2]
X_valid = X[N//2:3*N//4]
X_test = X[3*N//4:]
y_train = y[:N//2]
y_valid = y[N//2:3*N//4]
y_test = y[3*N//4:]

len(X), len(X_train), len(X_test)


# In[79]:


model = linear_model.LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)


# In[80]:


predictionsTrain = model.predict(X_train)
predictionsValid = model.predict(X_valid)
predictionsTest = model.predict(X_test)

correctPredictionsTrain = predictionsTrain == y_train
correctPredictionsValid = predictionsValid == y_valid
correctPredictionsTest = predictionsTest == y_test


# In[81]:


print("Accuracy of Train = " + str(sum(correctPredictionsTrain) / len(correctPredictionsTrain)))
print("Accuracy of Valid = " + str(sum(correctPredictionsValid) / len(correctPredictionsValid)))
print("Accuracy of Test = " + str(sum(correctPredictionsTest) / len(correctPredictionsTest)))


# In[82]:


def countBer(predictions, Y):
    TP = sum([(p and l) for (p,l) in zip(predictions, Y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, Y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, Y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, Y)])
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    #F1 = 2 * (precision*recall) / (precision + recall)
    #print(F1)
    #F10 = 101 * (precision*recall) / (100 * precision + recall)
    #print(F10)
    return 1 - 1/2 * (TPR + TNR)


# In[83]:


print("Balanced error rate of Train = " + str(countBer(predictionsTrain, y_train)))
print("Balanced error rate of Valid = " + str(countBer(predictionsValid, y_valid)))
print("Balanced error rate of Test = " + str(countBer(predictionsTest, y_test)))


# In[84]:


# Answer of Question 3:

# Accuracy of Train = 0.7947194719471947
# Balanced error rate of Train = 0.22465886939571145
# Accuracy of Valid = 0.7968337730870713
# Balanced error rate of Valid = 0.16536103542234337
# Accuracy of Test = 0.7770448548812665
# Balanced error rate of Test = 0.27657168701944823


# ### Question 4:

# In[86]:


def getBerNAccu(c, X, y):
    model = linear_model.LogisticRegression(C=c, class_weight='balanced')
    model.fit(X, y)
    predictions = model.predict(X)
    correct = predictions == y
    accu = sum(correct) / len(correct)
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    ber = 1 - 1/2 * (TPR + TNR)
    return ber, accu


# In[87]:


C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
berTrain = []
berValid = []
berTest = []
accuTrain = []
accuValid = []
accuTest = []
for c in C:
    ber_train, accu_train = getBerNAccu(c, X_train, y_train)
    berTrain.append(ber_train)
    accuTrain.append(accu_train)
    ber_valid, accu_valid = getBerNAccu(c, X_valid, y_valid)
    berValid.append(ber_valid)
    accuValid.append(accu_valid)
    ber_test, accu_test = getBerNAccu(c, X_test, y_test)
    berTest.append(ber_test)
    accuTest.append(accu_test)


# In[88]:


x = [10**i for i in range(-4, 5, 1)]
plt.loglog(x, berTrain, label = 'Train BER')
plt.loglog(x, berValid, label = 'Valid BER')
plt.loglog(x, berTest, label = 'Test BER')
plt.title("BER at Different C")
plt.xticks(x)
plt.xlabel("C")
plt.ylabel("BER")
plt.legend()


# In[89]:


x = [10**i for i in range(-4, 5, 1)]
plt.loglog(x, accuTrain, label = 'Train BER')
plt.loglog(x, accuValid, label = 'Valid BER')
plt.loglog(x, accuTest, label = 'Test BER')
plt.title("Accuracy at Different C")
plt.xticks(x)
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()


# In[90]:


# Answer to Question 4:
# BER at Different C is shown in the graph below
# I would choose 0.01 as my classifier, because the accuracy of 0.01
# is high and its BER is low comparably.


# ### Question 6:

# In[91]:


weights = [1.0] * len(y_train)
mod = linear_model.LogisticRegression(C=1, solver='lbfgs')
mod.fit(X_train, y_train, sample_weight=weights)


# In[92]:


def countTF(predictions, Y):
    TP = sum([(p and l) for (p,l) in zip(predictions, Y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, Y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, Y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, Y)])
    return TP, FP, TN, FN


# In[93]:


predictionsTest = model.predict(X_test)


# In[94]:


TP, FP, TN, FN = countTF(predictionsTest, y_test)


# In[95]:


precision = TP / (TP + FP)
recall = TP / (TP + FN)
precision, recall


# In[96]:


F1 = 2 * (precision*recall) / (precision + recall)
print("Unweighted F1 = " + str(F1))
F10 = 101 * (precision*recall) / (100 * (precision + recall))
print("Unweighted F10 = " + str(F10))


# In[97]:


weightPos = 1 - sum(d == True for d in y_train) / len(y_train)
weightNeg = 1 - weightPos

weights = [weightPos if i == True else weightNeg for i in y_train]
model = linear_model.LogisticRegression(C = 1, solver='lbfgs')
model.fit(X_train, y_train, sample_weight=weights);

predictionsTest = model.predict(X_test)
TP, FP, TN, FN = countTF(predictionsTest, y_test)

precision = TP / (TP + FP)
recall = TP / (TP + FN)

F1 = 2 * (precision*recall) / (precision + recall)
print("Weighted F1 = " + str(F1))
F10 = 101 * (precision*recall) / (100 * (precision + recall))
print("Weighted F10 = " + str(F10))


# In[ ]:


# Answer to Question 6:
# 


# ## Tasks - Diagnostics

# ### Question 7

# In[98]:


from sklearn.decomposition import PCA


# In[99]:


pca = PCA()
pca.fit(X_train)
print(pca.components_[0])


# ### Question 8

# In[100]:


def countComponent(X, y):
    berList = []
    for component in range(5, 31, 5):
        pca = PCA(n_components=component)
        pca.fit(X)
        Xpca = np.matmul(X, pca.components_.T)
        model = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
        model.fit(Xpca, y)
        predictions = model.predict(Xpca)
        berList.append(countBer(predictions, y))
    return berList


# In[101]:


ber_train = countComponent(X_train, y_train)
ber_valid = countComponent(X_valid, y_valid)
ber_test = countComponent(X_test, y_test)

plt.plot(range(5, 31, 5), ber_train, label='Train BER')
plt.plot(range(5, 31, 5), ber_valid, label='Valid BER')
plt.plot(range(5, 31, 5), ber_test, label='Test BER')
plt.title("BER of Collections")
plt.xlabel("Component")
plt.ylabel("BER")
plt.show()

