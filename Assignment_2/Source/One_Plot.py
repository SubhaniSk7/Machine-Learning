import pandas as pd
import math

import sklearn.linear_model
import sklearn.model_selection
# from sklearn import cross_validation
import random
import matplotlib.pyplot as mt
import numpy as np
from sklearn import preprocessing
import xlrd
import h5py

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
print('classes:', set(data1Y_Values))

mt.figure('data_1')
mt.scatter(data1X_Values[:, 0], data1X_Values[:, 1], c=data1Y_Values)
mt.xlabel('x1')
mt.ylabel('x2')
print('-------------------------------------------')

print(list(data2.keys()))
data2_X = data2['x']
data2_Y = data2['y']

data2X_Values = np.array(data2_X.value)
data2Y_Values = np.array(data2_Y.value)

print(data2X_Values.shape)
print(data2Y_Values.shape)
print('classes:', set(data2Y_Values))
mt.figure('data_2')
mt.scatter(data2X_Values[:, 0], data2X_Values[:, 1], c=data2Y_Values)
mt.xlabel('x1')
mt.ylabel('x2')
print('-------------------------------------------')

print(list(data3.keys()))
data3_X = data3['x']
data3_Y = data3['y']

data3X_Values = np.array(data3_X.value)
data3Y_Values = np.array(data3_Y.value)

print(data3X_Values.shape)
print(data3Y_Values.shape)

print('classes:', set(data3Y_Values))
mt.figure('data_3')
mt.scatter(data3X_Values[:, 0], data3X_Values[:, 1], c=data3Y_Values)
mt.xlabel('x1')
mt.ylabel('x2')
print('-------------------------------------------')

print(list(data4.keys()))
data4_X = data4['x']
data4_Y = data4['y']

data4X_Values = np.array(data4_X.value)
data4Y_Values = np.array(data4_Y.value)

print(data4X_Values.shape)
print(data4Y_Values.shape)

print('classes:', set(data4Y_Values))
mt.figure('data_4')
mt.scatter(data4X_Values[:, 0], data4X_Values[:, 1], c=data4Y_Values)
mt.xlabel('x1')
mt.ylabel('x2')

print('-------------------------------------------')

mt.show()
