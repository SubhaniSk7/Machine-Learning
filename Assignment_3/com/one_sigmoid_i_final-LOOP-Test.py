#!/usr/bin/env python
# coding: utf-8

# In[14]:


import copy
import numpy as np
from sklearn import preprocessing
import h5py
import sklearn.svm
import random
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mt


# In[15]:


class Layer:
    # constructor
    def __init__(self, neurons=0, theta=[], b=[], z=[], a=[],delta=[],DELTA_THETA=[],DELTA_BIAS=[],dTheta=[],dBias=[]):
        self.neurons = 0 # neurons count in layer
        self.theta = [] # Weight vector(W)
        self.b=[] # bias
        self.z = [] # hypothesis z = W.T * X + b = here = theta.T * X + b
        self.a = [] # activation function a=sigmoid(z) or relu(z) or anyother(z)
        self.delta = [] # Loss or Error function delta= delta_cross_entropy() or anyother()
        self.DELTA_THETA = [] # only derivative weight vector =dw
        self.DELTA_BIAS = [] # only derivative bias vector =db
        self.dTheta = [] # complete derivation term = (1/m)*(DELTA + (lambda*theta))
        self.dBias = [] # complete derivation term 

    def setNeurons(self, neurons):
        self.neurons = neurons

    def getNeurons(self):
        return self.neurons

    def setTheta(self, theta):
        self.theta = theta

    def getTheta(self):
        return self.theta

    def setB(self, b):
        self.b = b

    def getB(self):
        return self.b

    def setZ(self, z):
        self.z = z

    def getZ(self):
        return self.z

    def setA(self, a):
        self.a = a

    def getA(self):
        return self.a

    def setDelta(self, delta):
        self.delta = delta

    def getDelta(self):
        return self.delta

    def setDELTA_THETA(self, DELTA_THETA):
        self.DELTA_THETA = DELTA_THETA

    def getDELTA_THETA(self):
        return self.DELTA_THETA

    def setDELTA_BIAS(self, DELTA_BIAS):
        self.DELTA_BIAS = DELTA_BIAS

    def getDELTA_BIAS(self):
        return self.DELTA_BIAS
    
    def setDTheta(self, dTheta):
        self.dTheta = dTheta

    def getDTheta(self):
        return self.dTheta
    
    def setDBias(self, dBias):
        self.dBias = dBias

    def getDBias(self):
        return self.dBias
    
    


# In[16]:


dataset = h5py.File('../MNIST_Subset-4.h5', 'r+')

print(list(dataset.keys()))


# In[17]:


data_X = dataset['X']
data_Y = dataset['Y']

X = np.array(data_X.value)
Y = np.array(data_Y.value)

print(X.shape,Y.shape)

# Y=Y.reshape(14251,1)
print(X.shape,Y.shape)

print(X.shape)
X=X.reshape(14251,28*28)
print(X.shape)

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2,random_state=20)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

X=X_train #X= X_train
Y=Y_train #Y=Y_train

X = preprocessing.scale(X)
X_test=preprocessing.scale(X_test)

m=X.shape[0]
print('no.of samples:',m)

X=X.T # transposed X now shape=784 x 14251 => now each column is one datapoint
X_test=X_test.T
print(X.shape)


# In[18]:


def sigmoid(z):
    return (1.0/(1.0+np.exp(-z)))


# In[19]:


def softmax(z):
    return np.divide(np.exp(z),np.sum(np.exp(z),axis=0))


# In[20]:


def stableSoftmax(z):
    exps = np.exp(z - np.max(z))
    return np.divide(exps , np.sum(exps,axis=0))


# In[21]:


def delta_cross_entropy(z,y):
    grad = (z-y)/m
    return grad


# In[22]:


def crossEntropy(a,y):
    return (-y*np.log(a))/m


# In[23]:


def accuracy(y_predicted):
    y_multilabel = []
    for p in y_predicted:
        y_multilabel.append(list(p).index(max(p)))
    
    plot_accuracy.append(accuracy_score(y_multilabel, Y))
    print(accuracy_score(y_multilabel, Y))
    
def accuracy_test(y_predicted):
    y_multilabel = []
    for p in y_predicted:
        y_multilabel.append(list(p).index(max(p)))
        
    print(accuracy_score(y_multilabel, Y_test))


# In[24]:


y_actual=[] # changed to 1 at their index

for i in range(Y.shape[0]):
    temp = [0]*10
    index = int(Y[i])
    temp[index] = 1
    y_actual.append(temp)
y_actual=np.array(y_actual).T


# In[40]:


# l=int(input('enter total layers:'))
# layers=[]
# for i in range(l):
#     layers.append(int(input('enter neurons count:')))
# print(layers)

l=5
neurons=[784,100,50,50,10]

# l=3
# neurons=[784,100,10]

layers=[]
for i in range(len(neurons)):
    lay=Layer()
    
    if(i!=len(neurons)-1):
        DELTA_THETA=np.zeros((neurons[i+1],neurons[i]))
        theta=np.random.uniform(low=0.1,high=1,size=(neurons[i],neurons[i+1]))
        #DELTA_BIAS=
        #bias=
        
        lay.setDELTA_THETA(DELTA_THETA)
        lay.setTheta(theta)
        
    layers.append(lay)

plot_accuracy=[]

layers[0].setA(X)
regParam=0.0001
alpha=1
maxIterations=1000

for iter in range(maxIterations):
    
    # Forward propagation
    for i in range(1,l):
        z=np.dot(layers[i-1].getTheta().T, layers[i-1].getA())
        if(i==l-1):
            a=softmax(z)
        else:
            a=sigmoid(z)
        layers[i].setZ(z)
        layers[i].setA(a)
    
    # Backward Propagation
    for i in range(l-1,-1,-1):
        loss=None
        if(i==l-1):
            loss=delta_cross_entropy(layers[i].getA(),y_actual)
        else:
            loss=np.dot(layers[i].getTheta(),layers[i+1].getDelta()) * (layers[i].getA()*(1-layers[i].getA()))
        layers[i].setDelta(loss)
    
    for i in range(0,l-1):
        D=layers[i].getDELTA_THETA() + np.dot(layers[i+1].getDelta(),layers[i].getA().T)
        layers[i].setDELTA_THETA(D)
    
    for i in range(0,l-1):
        dT=(1/m)*(layers[i].getDELTA_THETA().T+(regParam*layers[i].getTheta()))
        layers[i].setDTheta(dT)
    
    print('Iteration:',iter,'--> ',end='')
    accuracy(layers[-1].getA().T)
    if(accuracy(layers[-1].getA().T) == np.nan):
        break
    
    for i in range(0,l-1):
        newTh=layers[i].getTheta()-(alpha*layers[i].getDTheta())
        layers[i].setTheta(newTh)


# In[42]:


mt.figure('Sigmoid 5 Layers')
mt.plot(range(len(plot_accuracy)), plot_accuracy, 'b', label='lambda=0.0001')

mt.xlabel('Iterations')
mt.ylabel('Accuracy')
mt.title('Sigmoid 5 Layers')
mt.axis('tight')
mt.show()


# In[43]:


layers_test=[]
for i in range(len(neurons)):
    lay=Layer()
    
    if(i!=len(neurons)-1):
        theta=layers[i].getTheta()
        lay.setTheta(theta)
    layers_test.append(lay)
    
layers_test[0].setA(X_test)
# Forward propagation
for i in range(1,l):
    z=np.dot(layers_test[i-1].getTheta().T, layers_test[i-1].getA())
    if(i==l-1):
        a=softmax(z)
    else:
        a=sigmoid(z)
        
    layers_test[i].setZ(z)
    layers_test[i].setA(a)

accuracy_test(layers_test[-1].getA().T)


# In[44]:


for i in range(len(neurons)-1):
    print(layers[i].getTheta())


# In[ ]:




