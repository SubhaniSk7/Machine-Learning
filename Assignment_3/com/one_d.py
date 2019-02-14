#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import numpy as np
from sklearn import preprocessing
import h5py
import sklearn.svm
import random
import math
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mt
from sklearn.neural_network import MLPClassifier


# In[2]:


dataset = h5py.File('../MNIST_Subset-3.h5', 'r+')

print(list(dataset.keys()))


# In[3]:


data_X = dataset['X']
data_Y = dataset['Y']

X = np.array(data_X.value)
Y = np.array(data_Y.value)

print(X.shape,Y.shape)

# Y=Y.reshape(14251,1)
print(X.shape,Y.shape)

print(X.shape)
X=X.reshape(14251,28*28)
# print(X.shape)

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2,random_state=20)
# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

X=X_train #X= X_train
Y=Y_train #Y=Y_train

X = preprocessing.scale(X)
X_test=preprocessing.scale(X_test)

m=X.shape[0]
# print('no.of samples:',m)


# In[ ]:





# In[4]:


model= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, ), random_state=1,activation='logistic')

model.fit(X,Y)


# In[5]:


predicted=model.predict(X)
print(predicted)
print(accuracy_score(Y,predicted))


# In[6]:


test_predicted=model.predict(X_test)
print(predicted)
print(accuracy_score(Y_test,test_predicted))


# In[ ]:





# In[7]:


model= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,50,50, ), random_state=1,activation='logistic')

model.fit(X,Y)


# In[8]:


predicted=model.predict(X)
print(predicted)
print(accuracy_score(Y,predicted))


# In[9]:


test_predicted=model.predict(X_test)
print(predicted)
print(accuracy_score(Y_test,test_predicted))


# In[ ]:





# In[10]:


model= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, ), random_state=1,activation='relu')

model.fit(X,Y)


# In[11]:


predicted=model.predict(X)
print(predicted)
print(accuracy_score(Y,predicted))


# In[12]:


test_predicted=model.predict(X_test)
print(predicted)
print(accuracy_score(Y_test,test_predicted))


# In[ ]:





# In[13]:


model= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,50,50, ), random_state=1,activation='relu')

model.fit(X,Y)


# In[14]:


predicted=model.predict(X)
print(predicted)
print(accuracy_score(Y,predicted))


# In[15]:


test_predicted=model.predict(X_test)
print(predicted)
print(accuracy_score(Y_test,test_predicted))


# In[ ]:





# In[ ]:




