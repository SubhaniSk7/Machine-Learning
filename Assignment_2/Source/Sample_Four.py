import copy

import numpy.linalg
import pandas as pd
import math

import sklearn.linear_model
import sklearn.model_selection
# from sklearn import cross_validation
import random
import matplotlib.pyplot as mt
import numpy as np
from sklearn import preprocessing, svm
import xlrd
import h5py
import cvxopt
import cvxopt.solvers
import sklearn.svm

from mlxtend.plotting import plot_decision_regions

data1 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_1.h5', 'r+')
data2 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_2.h5', 'r+')
data3 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_3.h5', 'r+')
data4 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_4.h5', 'r+')

print(list(data1.keys()))
data1_X = data1['x']
data1_Y = data1['y']

data1X_Values = np.array(data1_X.value)
data1Y_Values = np.array(data1_Y.value)
print(data1X_Values.shape)
print(data1Y_Values.shape)

print('y classes:', set(data1Y_Values))

mt.subplot(221)

model1 = svm.SVC(kernel='rbf')
model1.fit(data1X_Values, data1Y_Values)

print(model1)

print(model1.support_)
print(model1.support_vectors_)
print('support vectors choosen in classes:model1:', model1.n_support_)

print(data1X_Values.shape)
print(data1Y_Values.shape)
plot_decision_regions(data1X_Values, data1Y_Values, clf=model1, legend=2)

print('-------------------------------------------')

# ///////////////////////////////////////////////////////////////////////////////////////

print(list(data2.keys()))
data2_X = data2['x']
data2_Y = data2['y']

data2X_Values = np.array(data2_X.value)
data2Y_Values = np.array(data2_Y.value)

print(data2X_Values.shape)
print(data2Y_Values.shape)

print('y classes:', set(data2Y_Values))

mt.subplot(222)
# mt.scatter(pdata2_x1_0, pdata2_x2_0, marker='o')
# mt.scatter(pdata2_x1_1, pdata2_x2_1, marker='+')


model2 = svm.SVC(kernel='rbf')
model2.fit(data2X_Values, data2Y_Values)

print(model2)

print(model2.support_)
print(model2.support_vectors_)
print('support vectors choosen in classes:model2:', model2.n_support_)

# print(data2X_Values.shape)
# print(data2Y_Values.shape)
plot_decision_regions(data2X_Values, data2Y_Values, clf=model2, legend=2)

print('-------------------------------------------')

# ///////////////////////////////////////////////////////////////////////////////////////

print(list(data3.keys()))
data3_X = data3['x']
data3_Y = data3['y']

data3X_Values = np.array(data3_X.value)
data3Y_Values = np.array(data3_Y.value)

print(data3X_Values.shape)
print(data3Y_Values.shape)

print('y classes:', set(data3Y_Values))

mt.subplot(223)

model3 = svm.SVC(kernel='rbf')
model3.fit(data3X_Values, data3Y_Values)

print(model3)

print(model3.support_)
print(model3.support_vectors_)
print('support vectors choosen in classes: model3:', model3.n_support_)

plot_decision_regions(data3X_Values, data3Y_Values, clf=model3, legend=2)

print('-------------------------------------------')
# ///////////////////////////////////////////////////////////////////////////////////////

print(list(data4.keys()))
data4_X = data4['x']
data4_Y = data4['y']

data4X_Values = np.array(data4_X.value)
data4Y_Values = np.array(data4_Y.value)

print(data4X_Values.shape)
print(data4Y_Values.shape)

data4X_x1 = []
data4X_x2 = []
for i in data4X_Values:
    data4X_x1.append(i[0])
    data4X_x2.append(i[1])

print('y classes:', set(data4Y_Values))

mt.subplot(224)

model4 = svm.SVC(kernel='rbf')
model4.fit(data3X_Values, data3Y_Values)

print(model4)

print(model4.support_)
print(model4.support_vectors_)
print('support vectors choosen in classes: model4:', model4.n_support_)

plot_decision_regions(data4X_Values, data4Y_Values, clf=model4, legend=2)

print('-------------------------------------------')

mt.show()
