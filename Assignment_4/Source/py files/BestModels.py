#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
import math
from sklearn.metrics import f1_score, accuracy_score,roc_curve,auc
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import matplotlib.pyplot as mt

from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.externals import joblib


# In[2]:


filename='./german_credit.csv'
data = pd.read_csv(filename)

data.columns

Y=data['Creditability']
X=data.drop(['Creditability'],axis=1)

print(X.shape,Y.shape)

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2,random_state=27)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[5]:


decisionBestModel=joblib.load('./decisionTreeBestModel.joblib')
ytestPredicted=decisionBestModel.predict(X_test)
accuracy_score(Y_test,ytestPredicted)


# In[6]:


randomBestModel=joblib.load('./randomForestBestModel.joblib')
ytestPredicted=randomBestModel.predict(X_test)
accuracy_score(Y_test,ytestPredicted)


# In[ ]:




