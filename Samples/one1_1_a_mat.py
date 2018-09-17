import pandas as pd
import math

import sklearn.model_selection
from sklearn import cross_validation
import random
import matplotlib.pyplot as mt
import numpy as np
from sklearn import preprocessing
import scipy
import xlrd
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)


def gradient_descent(alpha, x, y, xV, yV, theta, maxIterations):
    converged = False
    iter = 0
    m = x.shape[1]
    print('m:', m)

    jHistory = [0] * maxIterations
    jHistoryValidation = [0] * maxIterations

    J = compute_cost_function(m, theta, x, y)
    print('J=', J)

    # x_transpose = np.transpose(x)
    y_transpose = np.transpose(y)

    while not converged:

        theta_transpose = np.transpose(theta)

        mul = np.matmul(theta_transpose, x)
        sub = np.subtract(mul, y_transpose)
        sigma = np.matmul(sub, np.transpose(x))

        cost = (1 / m) * sigma
        cost = np.transpose(cost)
        # print('cost:', cost.shape)

        theta = theta - (alpha * cost)
        # print('theta:', theta.shape)

        # mean squared error
        e = compute_cost_function(m, theta, x, y)
        f = compute_cost_function(m, theta, xV, yV)
        jHistory[iter] = e
        jHistoryValidation[iter] = f
        print(e, '-->', f)
        iter += 1

        if iter == maxIterations:
            print('Max interactions exceeded!')
            converged = True

    # return theta
    return jHistory, jHistoryValidation, theta


def compute_cost_function(m, theta, x, y):
    theta_transpose = np.transpose(theta)
    # print('x:', x.shape)
    # print('t:',x.shape)

    # print('theta_transpose:', theta_transpose.shape)
    mul = np.dot(theta_transpose, x)
    # print('mul:', mul.shape)
    # print('y:', y.shape)

    y = np.array(y).reshape(y.shape[0], 1)
    y_transpose = np.transpose(y)

    # print("y_transpose:", y_transpose.shape)

    sub = np.subtract(mul, y_transpose)
    # print('sub:', sub.shape)

    sq = np.power(sub, 2)
    # print('sq:', sq.shape)

    sigma = np.sum(sq)
    # print('sigma', sigma)

    cost = (1 / 2 / m) * sigma
    rmse = np.sqrt(cost)
    return rmse


# -----------------------------------------------------

def minMaxMean(x):
    xMean = []
    xMin = []
    xMax = []
    for i in range(x.shape[1]):
        col_values = [row[i] for row in x]
        # print(col_values)
        valueMean = np.mean(col_values)
        valueMin = min(col_values)
        valueMax = max(col_values)
        xMean.append(valueMean)
        xMin.append(valueMin)
        xMax.append(valueMax)
        # print('=======')

    xMean[0] = 0
    xMin[0] = 0
    xMax[0] = 1
    # print(xMean)
    # print(xMin)
    # print(xMax)
    return xMean, xMin, xMax


def normalize(xMean, xMin, xMax, x):
    # print(x[0])
    for row in x:
        for i in range(len(row)):
            row[i] = (row[i] - xMean[i]) / (xMax[i] - xMin[i])
            # print(row[i])
    # print(x[0])
    return x


data = pd.read_excel('/home/subhani007/Desktop/ML Assignment/boston.xls')
x = data.drop("MV", axis=1)
x.insert(0, 'x0', 1)
y = np.array(data['MV'])
y = y.reshape(y.shape[0], 1)

# x_scaled = preprocessing.normalize(x)
x_scaled = np.array(x)
print('x_scaled', x_scaled.shape)
# x_scaled=np.array(x_scaled).reshape(x.shape[0],x.shape[1])

# y_scaled = preprocessing.normalize(y)
y_scaled = y

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.20, random_state=0)

theta = np.zeros(x_train.shape[1], dtype='int').reshape(x_train.shape[1], 1)

m = x_train.shape[0]

xTrans = np.transpose(x_train)
print('Initial rmse:', compute_cost_function(m, theta, xTrans, y_train))
print('-----------------------------------')

maxIterations = 500

learning_rate1 = 0.9

# print(math.ceil(506 / 5), '\n', math.ceil(2 * 506 / 5), '\n', math.ceil(3 * 506 / 5), '\n', math.ceil(4 * 506 / 5),
#       '\n', math.ceil(5 * 506 / 5))

print(x_scaled.shape)

x1 = x_scaled[0:math.ceil(506 / 5), :]
x2 = x_scaled[math.ceil(506 / 5):math.ceil(2 * 506 / 5), :]
x3 = x_scaled[math.ceil(2 * 506 / 5):math.ceil(3 * 506 / 5), :]
x4 = x_scaled[math.ceil(3 * 506 / 5):math.ceil(4 * 506 / 5), :]
x5 = x_scaled[math.ceil(4 * 506 / 5):math.ceil(5 * 506 / 5), :]

# print(x1[0])
# print(x1[-1])
# print('///////////////')
# print(x2[0])
# print(x2[-1])
# print('///////////////')
#
# print(x3[0])
# print(x3[-1])
# print('///////////////')
# print(x4[0])
# print(x4[-1])
# print('///////////////')
#
# print(x5[0])
# print(x5[-1])
# print('///////////////')


y1 = y_scaled[0:math.ceil(506 / 5), :]
y2 = y_scaled[math.ceil(506 / 5):math.ceil(2 * 506 / 5), :]
y3 = y_scaled[math.ceil(2 * 506 / 5):math.ceil(3 * 506 / 5), :]
y4 = y_scaled[math.ceil(3 * 506 / 5):math.ceil(4 * 506 / 5), :]
y5 = y_scaled[math.ceil(4 * 506 / 5):math.ceil(5 * 506 / 5), :]

xFold1Train = np.concatenate((x1, x2, x3, x4), axis=0)
yFold1Train = np.concatenate((y1, y2, y3, y4), axis=0)
xFold1Validation = x5
yFold1Validation = y5
print(xFold1Train.shape, xFold1Validation.shape, yFold1Train.shape, yFold1Validation.shape)
print('----------')

xFold2Train = np.concatenate((x1, x2, x3, x5), axis=0)
yFold2Train = np.concatenate((y1, y2, y3, y5), axis=0)
xFold2Validation = x4
yFold2Validation = y4
print(xFold2Train.shape, xFold2Validation.shape, yFold2Train.shape, yFold2Validation.shape)
print('----------')

xFold3Train = np.concatenate((x1, x2, x4, x5), axis=0)
yFold3Train = np.concatenate((y1, y2, y4, y5), axis=0)
xFold3Validation = x3
yFold3Validation = y3
print(xFold3Train.shape, xFold3Validation.shape, yFold3Train.shape, yFold3Validation.shape)
print('----------')

xFold4Train = np.concatenate((x1, x3, x4, x5), axis=0)
yFold4Train = np.concatenate((y1, y3, y4, y5), axis=0)
xFold4Validation = x2
yFold4Validation = y2
print(xFold4Train.shape, xFold4Validation.shape, yFold4Train.shape, yFold4Validation.shape)
print('----------')

xFold5Train = np.concatenate((x2, x3, x4, x5), axis=0)
yFold5Train = np.concatenate((y2, y3, y4, y5), axis=0)
xFold5Validation = x1
yFold5Validation = y1
print(xFold5Train.shape, xFold5Validation.shape, yFold5Train.shape, yFold5Validation.shape)
print('----------')

mt.figure(figsize=(10, 5))
for i in range(0, 5):
    xTrain = []
    if (i == 0):
        xMean, xMin, xMax = minMaxMean(xFold1Train)

        # print(xFold1Train[0])
        x_normalized = normalize(xMean, xMin, xMax, xFold1Train)
        # xTrans = np.transpose(xFold1Train)
        xTrans = np.transpose(x_normalized)

        # print('=================')
        #
        # print(x_normalized[0])
        #
        print('=================')

        print(xFold1Validation[0])
        xValidation_normalized = normalize(xMean, xMin, xMax, xFold1Validation)

        print(xValidation_normalized[0])
        rmseFold1Train, rmseFold1Validation, ThetaFold1Train = gradient_descent(learning_rate1, xTrans, yFold1Train,
                                                                                np.transpose(xValidation_normalized),
                                                                                yFold1Validation, theta, maxIterations)

        print('------------------')
        print(ThetaFold1Train)

        xValidation = x5
        yValidation = y5

        # print('xValidation:', xValidation.shape)
        # print('yValidation:', yValidation.shape)
        mt.plot(range(np.array(rmseFold1Train).shape[0]), rmseFold1Train, 'r', label='Learning Rate=0.9')
        mt.plot(range(np.array(rmseFold1Validation).shape[0]), rmseFold1Validation, 'b', label='Learning Rate=0.9')
        print('==========')
    if (i == 1):
        xMean, xMin, xMax = minMaxMean(xFold2Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold2Train)

        # xTrans = np.transpose(xFold2Train)
        xTrans = np.transpose(x_normalized)

        rmseFold2Train, ThetaFold2Train = gradient_descent(learning_rate1, xTrans, yFold2Train, theta, maxIterations)

        mt.plot(np.arange(maxIterations), rmseFold2Train, 'g', label='Learning Rate=0.5')
        xValidation = x4
        yValidation = y4

        print('==========')
    if (i == 2):
        xMean, xMin, xMax = minMaxMean(xFold3Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold3Train)

        # xTrans = np.transpose(xFold3Train)
        xTrans = np.transpose(x_normalized)

        rmseFold3Train, ThetaFold3Train = gradient_descent(learning_rate1, xTrans, yFold3Train, theta, maxIterations)
        mt.plot(np.arange(maxIterations), rmseFold3Train, 'b', label='Learning Rate=0.5')
        xValidation = x3
        yValidation = y3
        print('==========')
    if (i == 3):
        xMean, xMin, xMax = minMaxMean(xFold4Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold4Train)

        # xTrans = np.transpose(xFold4Train)
        xTrans = np.transpose(x_normalized)

        rmseFold4Train, ThetaFold4Train = gradient_descent(learning_rate1, xTrans, yFold4Train, theta, maxIterations)
        mt.plot(np.arange(maxIterations), rmseFold4Train, 'c', label='Learning Rate=0.5')

        xValidation = x2
        yValidation = y2
        print('==========')
    if (i == 4):
        xMean, xMin, xMax = minMaxMean(xFold5Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold5Train)

        # xTrans = np.transpose(xFold5Train)
        xTrans = np.transpose(x_normalized)

        rmseFold5Train, ThetaFold5Train = gradient_descent(learning_rate1, xTrans, yFold5Train, theta, maxIterations)
        mt.plot(np.arange(maxIterations), rmseFold5Train, 'r', label='Learning Rate=0.5')

        xValidation = x1
        yValidation = y1
        print('==========')

#
# mt.plot(np.arange(maxIterations), Jhistory, 'r', label='Learning Rate=0.5')
#
mt.title('Linear Regression')
mt.ylabel('RMSE')
mt.xlabel('iterations')
#
mt.legend(loc='upper right')
#
mt.show()
