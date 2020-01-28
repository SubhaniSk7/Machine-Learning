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
from sklearn.ensemble import RandomForestClassifier


# In[2]:


filename='./german_credit.csv'
data = pd.read_csv(filename)

Y=data['Creditability']
X=data.drop(['Creditability'],axis=1)
print(X.shape,Y.shape)

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2,random_state=27)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[3]:


model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
# model.get_params(deep=True)
ytrainPredicted=model.predict(X_train)
ytestPredicted=model.predict(X_test)
print('trainAccuracy:',accuracy_score(Y_train,ytrainPredicted),' testAccuracy:',accuracy_score(Y_test,ytestPredicted))
print('trainF1:',f1_score(Y_train,ytrainPredicted),' testF1:',f1_score(Y_test,ytestPredicted))


# In[ ]:





# In[4]:


model=RandomForestClassifier()
model.fit(X_train,Y_train)

ytrainPredicted=model.predict(X_train)
ytestPredicted=model.predict(X_test)
print('trainAcc:',accuracy_score(Y_train,ytrainPredicted),' testAcc:',accuracy_score(Y_test,ytestPredicted))
print(f1_score(Y_train,ytrainPredicted),f1_score(Y_test,ytestPredicted))


# In[ ]:





# In[5]:


decision_scores=cross_val_score(model,X_train,Y_train,cv=10)
print(decision_scores)

mt.title('decision tree cross_val_score')
mt.plot(range(len(decision_scores)),decision_scores,label='cross_val_score')
mt.xlabel('folds')
mt.ylabel('cross_val_score')
mt.legend()
mt.show()


# In[6]:


random_scores=cross_val_score(model,X_train,Y_train,cv=10)
print(random_scores)

mt.title('random forest cross_val_score')
mt.plot(range(len(random_scores)),random_scores,label='cross_val_score')
mt.xlabel('folds')
mt.ylabel('cross_val_score')
mt.legend()
mt.show()


# In[ ]:





# In[7]:


kf = KFold(n_splits=5)
kf.get_n_splits(X)


# # Decision Tree Kfold

# In[8]:


x_num=X_train
dec_score=[]

for train_index, test_index in kf.split(x_num):
    X_tr, X_te = X_train.iloc[train_index], X_train.iloc[test_index]
    y_tr, y_te = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    model_k=DecisionTreeClassifier(max_depth=10,min_samples_leaf=0.1,min_samples_split=0.2,random_state=None)
    model_k.fit(X_tr,y_tr)
    
#     ytrainPredicted=model_k.predict(X_tr)
    ytestPredicted=model_k.predict(X_te)
    print('testAccuracy:',accuracy_score(y_te,ytestPredicted))
    print('testF1:',f1_score(y_te,ytestPredicted))
    
    dec_score.append(accuracy_score(y_te,ytestPredicted))
    print('----')

    
mt.title('decision tree kfold validation score')
mt.plot(range(len(dec_score)),dec_score,label='score')
mt.show()


# In[ ]:





# # Random Forest Kfold

# In[9]:


x_num=X_train
rand_score=[]

for train_index, test_index in kf.split(x_num):
    X_tr, X_te = X_train.iloc[train_index], X_train.iloc[test_index]
    y_tr, y_te = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    model_k=RandomForestClassifier(n_estimators=8,max_depth=11,max_features=11,min_samples_leaf=3,random_state=None)
    model_k.fit(X_tr,y_tr)
    
#     ytrainPredicted=model_k.predict(X_tr)
    ytestPredicted=model_k.predict(X_te)
    print('testAccuracy:',accuracy_score(y_te,ytestPredicted))
    print('testF1:',f1_score(y_te,ytestPredicted))
    
    rand_score.append(accuracy_score(y_te,ytestPredicted))
    print('----')

    
mt.title('Random forest kfold validation score')
mt.plot(range(len(rand_score)),rand_score,label='score')
mt.show()


# In[10]:


np.var(dec_score)


# In[11]:


np.var(rand_score)


# In[12]:


np.var(dec_score)>np.var(rand_score)


# # Both

# In[16]:


kf = KFold(n_splits=5)

x_num=X_train
rand_score=[]
dec_score=[]

for train_index, test_index in kf.split(x_num):
    X_tr, X_te = X_train.iloc[train_index], X_train.iloc[test_index]
    y_tr, y_te = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    
    model_k=DecisionTreeClassifier(max_depth=5,min_samples_leaf=0.1,min_samples_split=0.2,random_state=None)
    model_k.fit(X_tr,y_tr)
    ytestPredicted=model_k.predict(X_te)
    dec_score.append(accuracy_score(y_te,ytestPredicted))
    
    model_k=RandomForestClassifier(n_estimators=8,max_depth=11,max_features=11,min_samples_leaf=3,random_state=None)
    model_k.fit(X_tr,y_tr)
    ytestPredicted=model_k.predict(X_te)
    
    rand_score.append(accuracy_score(y_te,ytestPredicted))


# In[17]:


# print(np.var(dec_score),np.var(rand_score))
np.var(dec_score)>np.var(rand_score)


# In[ ]:




