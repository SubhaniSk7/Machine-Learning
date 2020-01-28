#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as mt

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.transforms.functional as F

import pickle
from PIL import Image


# In[15]:


def extractFromPickle(file): # extract data from Pickle file 
    with open(file, 'rb') as fo:
        rawData = pickle.load(fo, encoding='bytes')
    return rawData


# In[41]:


train_path = '../train_CIFAR.pickle' # train file
train_data = extractFromPickle(train_path)
test_path = '../test_CIFAR.pickle' # test file
test_data = extractFromPickle(test_path)


# In[59]:


CIFAR_X_train = train_data["X"]
CIFAR_Y_train = train_data["Y"]
CIFAR_X_test = test_data["X"]
CIFAR_Y_test = test_data["Y"]


# In[43]:


X_train = CIFAR_X_train
X_train = X_train.reshape(10000,3,32,32).transpose(0,2,3,1) # changing dimension
X_test = CIFAR_X_test
X_test = X_test.reshape(2000,3,32,32).transpose(0,2,3,1)
print(X_train.shape,X_test.shape)


# In[44]:


alexnet = models.alexnet(pretrained=True) # alexnet--> pretrained model
alexnet


# In[45]:


# transforming the images with standard parameters
apply_transform = transforms.Compose([
    transforms.Resize((227, 4096)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[46]:


# feature extraction --> vector
def getVector(image):
    t_img = Variable(apply_transform(image)).unsqueeze(0)
    output = alexnet.classifier[6](t_img).detach().numpy().ravel()
    return output


# In[47]:


# images for train loading
img = []
for i in range(100):
    img.append(Image.fromarray(np.uint8(X_train[i])))


# In[48]:


# getting feature vector
featureVector = []
for i in range(len(img)):
    featureVector.append((getVector(img[i])))

featureVector


# In[50]:


# selecting 10% of test images
testImg = []
for i in range(10):
    testImg.append(Image.fromarray(np.uint8(X_test[i])))


# In[51]:


# getting test images feature vector
testFeatures = []
for i in range(len(testImg)):
    testFeatures.append((getVector(testImg[i])))


# In[52]:


# PCA--> Principal Component Analysis
pca = PCA(n_components=2)
pcaTrain = pca.fit_transform(featureVector)
pcaTest = pca.fit_transform(testFeatures)


# In[66]:


# Training using Linear SVM
model = svm.SVC(kernel ='linear', C=1)

y_train_subset = []
for i in range(100):
    y_train_subset.append(CIFAR_Y_train[i])
    
model.fit(pcaTrain, y_train_subset)


# In[68]:


yPredicted = model.predict(pcaTest) # predicted values


# In[69]:


y_test_actual = []
for i in range(10):
    y_test_actual.append(CIFAR_Y_test[i])
accuracy_score(y_test_actual,yPredicted)


# In[70]:


#FP=false positive; TP=true positive; thresholds

FP, TP, thresholds = metrics.roc_curve(y_test_actual, yPredicted, pos_label=2)


# In[71]:


TP = [0]*len(TP)

roc_auc=auc(FP,TP) # roc_curve 


# In[72]:


confusion_matrix(y_test_actual, yPredicted) # constructing confusion matrix


# In[73]:


mt.plot(FP, TP, color='darkorange',
         label='ROC curve (area = %0.2f)' % roc_auc)
mt.plot([0, 1], [0, 1], color='navy', linestyle='--')

mt.xlim([0.0, 1.0])
mt.ylim([0.0, 1.05])
mt.ylabel('True-Positive')
mt.xlabel('False-Positive')

mt.title('Receiver operating characteristic example')
mt.legend(loc="upper right")

mt.show()


# In[ ]:




