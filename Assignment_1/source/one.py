import pandas as pd
import math

import sklearn.linear_model
import sklearn.model_selection
from sklearn import cross_validation
import random
import matplotlib.pyplot as mt
import numpy as np
from sklearn import preprocessing
import scipy
import xlrd

np.set_printoptions(suppress=True)


# -----------------------------------------------------

def minMaxMean(x):
    xMean = []
    xMin = []
    xMax = []
    for i in range(x.shape[1]):
        col_values = [row[i] for row in x]
        # print(col_values)
        valueMean = np.mean(col_values)
        valueMin = np.min(col_values)
        valueMax = max(col_values)
        xMean.append(valueMean)
        xMin.append(valueMin)
        xMax.append(valueMax)

    xMean[0] = 0
    xMin[0] = 0
    xMax[0] = 1
    return xMean, xMin, xMax



def normalize(xMean, xMin, xMax, x):
    # print(x[0])
    for row in x:
        for i in range(len(row)):
            row[i] = (row[i] - xMean[i]) / (xMax[i] - xMin[i])
            # print(row[i])
    # print(x[0])
    return x


# ----------------------------------------------------------------------------

def gradientDescent(alpha, x, y, xV, yV, theta, maxIterations, norm, hyper):
    iter = 0
    examples = x.shape[1]
    print('Examples:', examples)

    rmseTrainHistory = [0] * maxIterations
    rmseValidationHistory = [0] * maxIterations

    # x_transpose = np.transpose(x)
    y_transpose = np.transpose(y)

    for i in range(maxIterations):

        theta_transpose = np.transpose(theta)

        mul = np.matmul(theta_transpose, x)
        sub = np.subtract(mul, y_transpose)
        sigma = np.matmul(sub, np.transpose(x))

        cost = sigma
        cost = np.transpose(cost)
        # print('cost:', cost.shape)

        # mean squared error
        trainRMSE = calculateRMSE(examples, theta, x, y)
        validationRMSE = calculateRMSE(examples, theta, xV, yV)

        if (norm == 'l0'):
            theta = theta - (alpha * cost) / examples
        if (norm == 'l1'):
            print('----------------------->l1')
            theta = theta - (alpha * (cost + (hyper / 2)) / examples)
        if (norm == 'l2'):
            theta = theta - (alpha * (cost + (hyper * theta)) / examples)
        # print('-->theta:', theta)

        rmseTrainHistory[iter] = trainRMSE
        rmseValidationHistory[iter] = validationRMSE
        print(trainRMSE, '-->', validationRMSE)
        iter += 1

    return rmseTrainHistory, rmseValidationHistory, theta


# ----------------------------------------------------------------------------

def calculateRMSE(examples, theta, x, y):
    theta_transpose = np.transpose(theta)
    mul = np.dot(theta_transpose, x)
    y = np.array(y).reshape(y.shape[0], 1)
    y_transpose = np.transpose(y)
    sub = mul - y_transpose
    sq = np.power(sub, 2)
    sigma = np.sum(sq)

    cost = (1 / 2 / examples) * sigma
    rmse = np.sqrt(cost)
    return rmse


# ----------------------------------------------------------------------------

#data = pd.read_excel('/home/subhani007/Desktop/ML Assignment/boston.xls')
data = pd.read_excel('./boston.xls')
x = data.drop("MV", axis=1)
x.insert(0, 'x0', 1)
y = np.array(data['MV'])
y = y.reshape(y.shape[0], 1)
# print('`````````````',y[0])

x_scaled = np.array(x)
print('x_scaled', x_scaled.shape)
# x_scaled=np.array(x_scaled).reshape(x.shape[0],x.shape[1])

y_scaled = np.array(y)

theta = np.zeros(x_scaled.shape[1], dtype='int').reshape(x_scaled.shape[1], 1)
# print('_______________theta:')
# print(theta)
# m = x_train.shape[0]
m = x_scaled.shape[0]

# xTrans = np.transpose(x_train)
# print('Initial rmse:', compute_cost_function(m, theta, xTrans, y_train))
print('-----------------------------------')

maxIterations = 500

learning_rate = 0.9

# print(math.ceil(506 / 5), '\n', math.ceil(2 * 506 / 5), '\n', math.ceil(3 * 506 / 5), '\n', math.ceil(4 * 506 / 5),
#       '\n', math.ceil(5 * 506 / 5))

print(x_scaled.shape)

x1 = x_scaled[0:math.ceil(506 / 5), :]
x2 = x_scaled[math.ceil(506 / 5):math.ceil(2 * 506 / 5), :]
x3 = x_scaled[math.ceil(2 * 506 / 5):math.ceil(3 * 506 / 5), :]
x4 = x_scaled[math.ceil(3 * 506 / 5):math.ceil(4 * 506 / 5), :]
x5 = x_scaled[math.ceil(4 * 506 / 5):math.ceil(5 * 506 / 5), :]

y1 = y_scaled[0:math.ceil(506 / 5), :]
y2 = y_scaled[math.ceil(506 / 5):math.ceil(2 * 506 / 5), :]
y3 = y_scaled[math.ceil(2 * 506 / 5):math.ceil(3 * 506 / 5), :]
y4 = y_scaled[math.ceil(3 * 506 / 5):math.ceil(4 * 506 / 5), :]
y5 = y_scaled[math.ceil(4 * 506 / 5):math.ceil(5 * 506 / 5), :]

xFold1Train = np.concatenate((x1, x2, x3, x4), axis=0)
yFold1Train = np.concatenate((y1, y2, y3, y4), axis=0)
xFold1Validation = x5
yFold1Validation = y5
# print(xFold1Train.shape, xFold1Validation.shape, yFold1Train.shape, yFold1Validation.shape)
# print('----------')

xFold2Train = np.concatenate((x1, x2, x3, x5), axis=0)
yFold2Train = np.concatenate((y1, y2, y3, y5), axis=0)
xFold2Validation = x4
yFold2Validation = y4
# print(xFold2Train.shape, xFold2Validation.shape, yFold2Train.shape, yFold2Validation.shape)
# print('----------')

xFold3Train = np.concatenate((x1, x2, x4, x5), axis=0)
yFold3Train = np.concatenate((y1, y2, y4, y5), axis=0)
xFold3Validation = x3
yFold3Validation = y3
# print(xFold3Train.shape, xFold3Validation.shape, yFold3Train.shape, yFold3Validation.shape)
# print('----------')

xFold4Train = np.concatenate((x1, x3, x4, x5), axis=0)
yFold4Train = np.concatenate((y1, y3, y4, y5), axis=0)
xFold4Validation = x2
yFold4Validation = y2
# print(xFold4Train.shape, xFold4Validation.shape, yFold4Train.shape, yFold4Validation.shape)
# print('----------')

xFold5Train = np.concatenate((x2, x3, x4, x5), axis=0)
yFold5Train = np.concatenate((y2, y3, y4, y5), axis=0)
xFold5Validation = x1
yFold5Validation = y1
# print(xFold5Train.shape, xFold5Validation.shape, yFold5Train.shape, yFold5Validation.shape)
# print('----------')

for i in range(0, 5):
    # print('_______________theta:')
    # print(theta)
    if (i == 0):
        xMean, xMin, xMax = minMaxMean(xFold1Train)

        # print(xFold1Train[0])
        x_normalized = normalize(xMean, xMin, xMax, xFold1Train)
        # xTrans = np.transpose(xFold1Train)
        xTrans1 = np.transpose(x_normalized)

        print('=================')

        print(xFold1Validation[0])
        xValidation_normalized = normalize(xMean, xMin, xMax, xFold1Validation)

        print(xValidation_normalized[0])
        rmseFold1Train, rmseFold1Validation, ThetaFold1Train = gradientDescent(learning_rate, xTrans1, yFold1Train,
                                                                               np.transpose(xValidation_normalized),
                                                                               yFold1Validation, theta, maxIterations,
                                                                               'l0', 0)

        print('------------------')
        print('-->', ThetaFold1Train)

        mt.figure('Fold1')
        # mt.subplot(1, 2, 1)
        mt.plot(range(np.array(rmseFold1Train).shape[0]), rmseFold1Train, 'r', label='RMSE of Train')
        mt.plot(range(np.array(rmseFold1Validation).shape[0]), rmseFold1Validation, 'b', label='RMSE of Validation')

        mt.title('Fold1 Linear Regression')
        mt.ylabel('RMSE')
        mt.xlabel('iterations')
        mt.legend(loc='upper right')
        print('==========')
    if (i == 1):
        xMean, xMin, xMax = minMaxMean(xFold2Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold2Train)

        # xTrans = np.transpose(xFold2Train)
        xTrans2 = np.transpose(x_normalized)

        print(xFold2Validation[0])
        xValidation_normalized = normalize(xMean, xMin, xMax, xFold2Validation)

        print(xValidation_normalized[0])
        rmseFold2Train, rmseFold2Validation, ThetaFold2Train = gradientDescent(learning_rate, xTrans2, yFold2Train,
                                                                               np.transpose(xValidation_normalized),
                                                                               yFold2Validation, theta, maxIterations,
                                                                               'l0', 0)

        print('------------------')
        print(ThetaFold2Train)

        mt.figure('Fold2')
        # mt.subplot(1, 2, 2)
        mt.plot(range(np.array(rmseFold2Train).shape[0]), rmseFold2Train, 'r', label='RMSE of Train')
        mt.plot(range(np.array(rmseFold2Validation).shape[0]), rmseFold2Validation, 'b', label='RMSE of Validation')

        mt.title('Fold2 Linear Regression')
        mt.ylabel('RMSE')
        mt.xlabel('iterations')
        mt.legend(loc='upper right')
        print('==========')
    if (i == 2):
        xMean, xMin, xMax = minMaxMean(xFold3Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold3Train)

        # xTrans = np.transpose(xFold3Train)
        xTrans3 = np.transpose(x_normalized)

        print(xFold3Validation[0])
        xValidation_normalized = normalize(xMean, xMin, xMax, xFold3Validation)

        print(xValidation_normalized[0])
        rmseFold3Train, rmseFold3Validation, ThetaFold3Train = gradientDescent(learning_rate, xTrans3, yFold3Train,
                                                                               np.transpose(xValidation_normalized),
                                                                               yFold3Validation, theta, maxIterations,
                                                                               'l0', 0)

        print('------------------')
        print(ThetaFold3Train)

        mt.figure('Fold3')
        # mt.subplot(5, 2, 3)
        mt.plot(range(np.array(rmseFold3Train).shape[0]), rmseFold3Train, 'r', label='RMSE of Train')
        mt.plot(range(np.array(rmseFold3Validation).shape[0]), rmseFold3Validation, 'b', label='RMSE of Validation')

        mt.title('Fold3 Linear Regression')
        mt.ylabel('RMSE')
        mt.xlabel('iterations')
        mt.legend(loc='upper right')
        print('==========')
    if (i == 3):
        xMean, xMin, xMax = minMaxMean(xFold4Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold4Train)

        # xTrans = np.transpose(xFold4Train)
        xTrans4 = np.transpose(x_normalized)

        print(xFold4Validation[0])
        xValidation_normalized = normalize(xMean, xMin, xMax, xFold4Validation)

        print(xValidation_normalized[0])
        rmseFold4Train, rmseFold4Validation, ThetaFold4Train = gradientDescent(learning_rate, xTrans4, yFold4Train,
                                                                               np.transpose(xValidation_normalized),
                                                                               yFold4Validation, theta, maxIterations,
                                                                               'l0', 0)

        print('------------------')
        print(ThetaFold4Train)

        mt.figure('Fold4')
        # mt.subplot(5, 2, 4)
        mt.plot(range(np.array(rmseFold4Train).shape[0]), rmseFold4Train, 'r', label='RMSE of Train')
        mt.plot(range(np.array(rmseFold4Validation).shape[0]), rmseFold4Validation, 'b', label='RMSE of Validation')

        mt.title('Fold4 Linear Regression')
        mt.ylabel('RMSE')
        mt.xlabel('iterations')
        mt.legend(loc='upper right')
        print('==========')
    if (i == 4):
        print('//////////////', yFold5Validation[0])
        xMean, xMin, xMax = minMaxMean(xFold5Train)
        x_normalized = normalize(xMean, xMin, xMax, xFold5Train)

        # xTrans = np.transpose(xFold4Train)
        xTrans5 = np.transpose(x_normalized)

        print(xFold5Validation[0])
        xValidation_normalized = normalize(xMean, xMin, xMax, xFold5Validation)

        print(xValidation_normalized[0])
        rmseFold5Train, rmseFold5Validation, ThetaFold5Train = gradientDescent(learning_rate, xTrans5, yFold5Train,
                                                                               np.transpose(xValidation_normalized),
                                                                               yFold5Validation, theta, maxIterations,
                                                                               'l0', 0)

        print('------------------')
        print(ThetaFold5Train)

        mt.figure('Fold5')
        # mt.subplot(5, 2, 5)
        mt.plot(range(np.array(rmseFold5Train).shape[0]), rmseFold5Train, 'r', label='RMSE of Train')
        mt.plot(range(np.array(rmseFold5Validation).shape[0]), rmseFold5Validation, 'b', label='RMSE of Validation')

        mt.title('Fold5 Linear Regression')
        mt.ylabel('RMSE')
        mt.xlabel('iterations')
        mt.legend(loc='upper right')
        print('==========')

meanTrainRMSE = []
meanValidationRMSE = []
sdTrain = []
sdValidation = []

for i in range(maxIterations):
    rmseMeanTrain = (rmseFold1Train[i] + rmseFold2Train[i] + rmseFold3Train[i] + rmseFold4Train[i] + rmseFold5Train[
        i]) / 5

    meanTrainRMSE.append(rmseMeanTrain)

    rmseMeanValidation = (rmseFold1Validation[i] + rmseFold2Validation[i] + rmseFold3Validation[i] +
                          rmseFold4Validation[i] +
                          rmseFold5Validation[
                              i]) / 5

    meanValidationRMSE.append(rmseMeanValidation)

    sdTemp1 = []
    sdTemp1.append(rmseFold1Validation[i])
    sdTemp1.append(rmseFold2Validation[i])
    sdTemp1.append(rmseFold3Validation[i])
    sdTemp1.append(rmseFold4Validation[i])
    sdTemp1.append(rmseFold5Validation[i])

    sdTemp2 = []
    sdTemp2.append(rmseFold1Train[i])
    sdTemp2.append(rmseFold2Train[i])
    sdTemp2.append(rmseFold3Train[i])
    sdTemp2.append(rmseFold4Train[i])
    sdTemp2.append(rmseFold5Train[i])

    sdValidation.append(np.std(sdTemp1, axis=0))
    sdTrain.append(np.std(sdTemp2, axis=0))

mt.figure('MeanRMSE')
mt.plot(range(len(meanTrainRMSE)), meanTrainRMSE, 'r', label='Mean RMSE of Train')
mt.plot(range(len(meanValidationRMSE)), meanValidationRMSE, 'b', label='Mean RMSE of Validation')
mt.title('Mean RMSE')
mt.ylabel('MEAN RMSE')
mt.xlabel('iterations')
mt.legend(loc='upper right')

mt.figure('STD')
mt.plot(range(len(sdTrain)), sdTrain, 'r', label='STD of Train')
mt.plot(range(len(sdValidation)), sdValidation, 'b', label='STD of Validation')
mt.title('Standard Deviation')
mt.ylabel('STD')
mt.xlabel('iterations')
mt.legend(loc='upper right')

mt.figure('STD Bar')
mt.bar(range(len(sdTrain)), sdTrain, color='b', label='STD of Train')
mt.plot(range(len(sdValidation)), sdValidation, color='r', label='STD of Validation')
# mt.plot(range(len(meanTrainRMSE)), meanTrainRMSE, 'r', label='Mean RMSE of Train')

mt.title('Standard Deviation')
mt.ylabel('STD')
mt.xlabel('iterations')
mt.legend(loc='upper right')

min = math.inf
index = -1
toRegular = []
toRegular.append(rmseFold1Validation[-1])  # fold1
toRegular.append(rmseFold2Validation[-1])  # fold2
toRegular.append(rmseFold3Validation[-1])  # fold3
toRegular.append(rmseFold4Validation[-1])  # fold4
toRegular.append(rmseFold5Validation[-1])  # fold5

for i in range(len(toRegular)):
    print('toRegular:', i, '-->', toRegular[i])
    if (min > toRegular[i]):
        min = toRegular[i]
        index = i
print(index)

if (index == 0):
    xTestToRegular = xFold1Validation
    yTestToRegular = yFold1Validation
    xTrainToRegular = xFold1Train
    yTrainToRegular = yFold1Train


elif (index == 1):
    xTestToRegular = xFold2Validation
    yTestToRegular = yFold2Validation
    xTrainToRegular = xFold2Train
    yTrainToRegular = yFold2Train
elif (index == 2):
    xTestToRegular = xFold3Validation
    yTestToRegular = yFold3Validation
    xTrainToRegular = xFold3Train
    yTrainToRegular = yFold3Train
elif (index == 3):
    xTestToRegular = xFold4Validation
    yTestToRegular = yFold4Validation
    xTrainToRegular = xFold4Train
    yTrainToRegular = yFold4Train
elif (index == 4):
    xTestToRegular = xFold5Validation
    yTestToRegular = yFold5Validation
    xTrainToRegular = xFold5Train
    yTrainToRegular = yFold5Train

# -------------------------------------------------------------------
parameters = {'alpha': [0.8, 0.9, 0.1, 0.2], 'normalize': [True, False]}
#
p = sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(), parameters, cv=5)
#
#

print()
xMean, xMin, xMax = minMaxMean(xTrainToRegular)
x_normalized = normalize(xMean, xMin, xMax, xTrainToRegular)

xTransRegular = np.transpose(x_normalized)

p.fit(xTrainToRegular, yTrainToRegular)
hyperLambda = p.best_params_['alpha']

print('------------------------>', hyperLambda)
xValidation_normalized = normalize(xMean, xMin, xMax, xTestToRegular)

# print(theta)

rmseTrainRegularL1, rmseValidationRegularL1, ThetaTrainRegularL1 = gradientDescent(learning_rate,
                                                                                   xTransRegular,
                                                                                   yTrainToRegular,
                                                                                   np.transpose(
                                                                                       xValidation_normalized),
                                                                                   yTestToRegular, theta,
                                                                                   maxIterations,
                                                                                   'l1', hyperLambda)
mt.figure('L1 Regularization')
mt.plot(range(np.array(rmseTrainRegularL1).shape[0]), rmseTrainRegularL1, 'r', label='RMSE of Train')
mt.plot(range(np.array(rmseValidationRegularL1).shape[0]), rmseValidationRegularL1, 'b', label='RMSE of Validation')
mt.title('L1 Regularization')
mt.ylabel('RMSE')
mt.xlabel('iterations')
mt.legend(loc='upper right')


parameters = {'alpha': [0.8, 0.9, 0.1, 0.2], 'normalize': [True, False]}
#
p = sklearn.model_selection.GridSearchCV(sklearn.linear_model.Ridge(), parameters, cv=5)
#
#
print()
xMean, xMin, xMax = minMaxMean(xTrainToRegular)
x_normalized = normalize(xMean, xMin, xMax, xTrainToRegular)

xTransRegular = np.transpose(x_normalized)

p.fit(xTrainToRegular, yTrainToRegular)
hyperLambda = p.best_params_['alpha']

print('------------------------>', hyperLambda)
xValidation_normalized = normalize(xMean, xMin, xMax, xTestToRegular)

rmseTrainRegularL2, rmseValidationRegularL2, ThetaTrainRegularL2 = gradientDescent(learning_rate,
                                                                                   xTransRegular,
                                                                                   yTrainToRegular,
                                                                                   np.transpose(
                                                                                       xValidation_normalized),
                                                                                   yTestToRegular, theta,
                                                                                   maxIterations,
                                                                                   'l2', hyperLambda)
mt.figure('L2 Regularization')
mt.plot(range(np.array(rmseTrainRegularL2).shape[0]), rmseTrainRegularL2, 'r', label='RMSE of Train')
mt.plot(range(np.array(rmseValidationRegularL2).shape[0]), rmseValidationRegularL2, 'b', label='RMSE of Validation')
mt.title('L2 Regularization')
mt.ylabel('RMSE')
mt.xlabel('iterations')
mt.legend(loc='upper right')

mt.show()
