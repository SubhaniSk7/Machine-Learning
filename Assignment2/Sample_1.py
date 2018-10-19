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

d1 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_1.h5', 'r+')
d2 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_2.h5', 'r+')
d3 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_3.h5', 'r+')
d4 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_4.h5', 'r+')

print(list(d1.keys()))
dx1 = d1['x']
y1 = d1['y']
x1_Values = np.array(dx1.value)
y1_Values = np.array(y1.value)
print(x1_Values.shape)
y1_Values = y1_Values.reshape(y1_Values.shape[0], 1)
print(y1_Values.shape)

dx1 = []
dx2 = []
for i in x1_Values:
    dx1.append(i[0])
    dx2.append(i[1])

d1_a0 = np.where(y1_Values == 0)
d1_a1 = np.where(y1_Values == 1)

d1_a0 = list(d1_a0[0])
d1_a1 = list(d1_a1[0])

px1_0 = []
px2_0 = []
px1_1 = []
px2_1 = []
for i in range(len(d1_a0)):
    px1_0.append(dx1[d1_a0[i]])
    px2_0.append(dx2[d1_a0[i]])
for i in range(len(d1_a1)):
    px1_1.append(dx1[d1_a1[i]])
    px2_1.append(dx2[d1_a1[i]])

mt.scatter(px1_0, px2_0, marker='o')
mt.scatter(px1_1, px2_1, marker='+')

mt.show()
print('--------------------------')
