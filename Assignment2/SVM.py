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

data1X_x1 = []
data1X_x2 = []
for i in data1X_Values:
    data1X_x1.append(i[0])
    data1X_x2.append(i[1])

data1Y_0 = np.where(data1Y_Values == 0)  # getting indices where y=0
data1Y_1 = np.where(data1Y_Values == 1)  # getting indices where y=1

data1Y_0 = list(data1Y_0[0])
data1Y_1 = list(data1Y_1[0])
# print(data1Y_0)
# print(data1Y_1)

pdata1_x1_0 = []  # plot data1_ x1 where y=0
pdata1_x2_0 = []  # plot data1_ x2 where y=0
pdata1_x1_1 = []  # plot data1_ x1 where y=1
pdata1_x2_1 = []  # plot data1_ x1 where y=1
for i in data1Y_0:
    x1 = data1X_x1[i]
    x2 = data1X_x2[i]
    pdata1_x1_0.append(x1)
    pdata1_x2_0.append(x2)

for i in data1Y_1:
    # print(i)
    x1 = data1X_x1[i]
    x2 = data1X_x2[i]
    # print(x1, '   ', x2)
    pdata1_x1_1.append(x1)
    pdata1_x2_1.append(x2)

print(set(data1Y_Values))
print(len(pdata1_x1_0))
print(len(pdata1_x1_1))

mt.subplot(221)
# mt.scatter(pdata1_x1_0, pdata1_x2_0, marker='o')
# mt.scatter(pdata1_x1_1, pdata1_x2_1, marker='+')
mt.scatter(data1X_x1, data1X_x2, marker='+', c=data1Y_Values)

print('-------------------------------------------')

print(list(data2.keys()))
data2_X = data2['x']
data2_Y = data2['y']

data2X_Values = np.array(data2_X.value)
data2Y_Values = np.array(data2_Y.value)

print(data2X_Values.shape)
print(data2Y_Values.shape)

data2X_x1 = []
data2X_x2 = []
for i in data2X_Values:
    data2X_x1.append(i[0])
    data2X_x2.append(i[1])

data2Y_0 = np.where(data2Y_Values == 0)  # getting indices where y=0
data2Y_1 = np.where(data2Y_Values == 1)  # getting indices where y=1

data2Y_0 = list(data2Y_0[0])
data2Y_1 = list(data2Y_1[0])

pdata2_x1_0 = []  # plot data2_ x1 where y=0
pdata2_x2_0 = []  # plot data2_ x2 where y=0
pdata2_x1_1 = []  # plot data2_ x1 where y=1
pdata2_x2_1 = []  # plot data2_ x1 where y=1
for i in data2Y_0:
    x1 = data2X_x1[i]
    x2 = data2X_x2[i]
    pdata2_x1_0.append(x1)
    pdata2_x2_0.append(x2)

for i in data2Y_1:
    # print(i)
    x1 = data2X_x1[i]
    x2 = data2X_x2[i]
    pdata2_x1_1.append(x1)
    pdata2_x2_1.append(x2)

print(set(data2Y_Values))
print(len(pdata2_x1_0))
print(len(pdata2_x1_1))

mt.subplot(222)
# mt.scatter(pdata2_x1_0, pdata2_x2_0, marker='o')
# mt.scatter(pdata2_x1_1, pdata2_x2_1, marker='+')
mt.scatter(data2X_x1, data2X_x2, marker='+', c=data2Y_Values)

print('-------------------------------------------')

print(list(data3.keys()))
data3_X = data3['x']
data3_Y = data3['y']

data3X_Values = np.array(data3_X.value)
data3Y_Values = np.array(data3_Y.value)

print(data3X_Values.shape)
print(data3Y_Values.shape)

data3X_x1 = []
data3X_x2 = []
for i in data3X_Values:
    data3X_x1.append(i[0])
    data3X_x2.append(i[1])

data3Y_0 = np.where(data3Y_Values == 0)  # getting indices where y=0
data3Y_1 = np.where(data3Y_Values == 1)  # getting indices where y=1
data3Y_2 = np.where(data3Y_Values == 2)  # getting indices where y=1

data3Y_0 = list(data3Y_0[0])
data3Y_1 = list(data3Y_1[0])
data3Y_2 = list(data3Y_2[0])

pdata3_x1_0 = []  # plot data3_ x1 where y=0
pdata3_x2_0 = []  # plot data3_ x2 where y=0
pdata3_x1_1 = []  # plot data3_ x1 where y=1
pdata3_x2_1 = []  # plot data3_ x1 where y=1
for i in data3Y_0:
    x1 = data3X_x1[i]
    x2 = data3X_x2[i]
    pdata3_x1_0.append(x1)
    pdata3_x2_0.append(x2)

for i in data3Y_1:
    x1 = data3X_x1[i]
    x2 = data3X_x2[i]
    pdata3_x1_1.append(x1)
    pdata3_x2_1.append(x2)

for i in data3Y_2:
    x1 = data3X_x1[i]
    x2 = data3X_x2[i]
    pdata3_x1_1.append(x1)
    pdata3_x2_1.append(x2)

print(set(data3Y_Values))
print(len(pdata3_x1_0))
print(len(pdata3_x1_1))

mt.subplot(223)
# mt.scatter(pdata3_x1_0, pdata3_x2_0, marker='o')
# mt.scatter(pdata3_x1_1, pdata3_x2_1, marker='+')
mt.scatter(data3X_x1, data3X_x2, marker='+', c=data3Y_Values)

print('-------------------------------------------')

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

print(set(data4Y_Values))
data4Y_0 = np.where(data4Y_Values == 0)  # getting indices where y=0
data4Y_1 = np.where(data4Y_Values == 1)  # getting indices where y=1
data4Y_2 = np.where(data4Y_Values == 2)  # getting indices where y=1

data4Y_0 = list(data4Y_0[0])
data4Y_1 = list(data4Y_1[0])
data4Y_2 = list(data4Y_2[0])

pdata4_x1_0 = []  # plot data4_ x1 where y=0
pdata4_x2_0 = []  # plot data4_ x2 where y=0
pdata4_x1_1 = []  # plot data4_ x1 where y=1
pdata4_x2_1 = []  # plot data4_ x1 where y=1
for i in data4Y_0:
    x1 = data4X_x1[i]
    x2 = data4X_x2[i]
    pdata4_x1_0.append(x1)
    pdata4_x2_0.append(x2)

for i in data4Y_1:
    x1 = data4X_x1[i]
    x2 = data4X_x2[i]
    pdata4_x1_1.append(x1)
    pdata4_x2_1.append(x2)

for i in data4Y_2:
    x1 = data4X_x1[i]
    x2 = data4X_x2[i]
    pdata4_x1_1.append(x1)
    pdata4_x2_1.append(x2)

mt.subplot(224)

print(data4Y_Values)
print(len(pdata4_x1_0))
print(len(pdata4_x1_1))
# mt.scatter(pdata4_x1_0, pdata4_x2_0, marker='o')
# mt.scatter(pdata4_x1_1, pdata4_x2_1, marker='+')
mt.scatter(data4X_x1, data4X_x2, marker='+', c=data4Y_Values)

print('-------------------------------------------')

mt.show()
