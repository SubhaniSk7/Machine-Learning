import copy

import h5py
import mlxtend.plotting
import sklearn.metrics
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn import metrics
import numpy as np
import sklearn.svm
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as mt

data3 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_3.h5', 'r+')


def predict():
    pass


data3_X = data3['x']
data3_Y = data3['y']

data3X_Values = np.array(data3_X.value)
data3Y_Values = np.array(data3_Y.value)

print(data3X_Values.shape)
print(data3Y_Values.shape)
data3X_train, data3X_test, data3Y_train, data3Y_test = train_test_split(data3X_Values, data3Y_Values, test_size=0.2,
                                                                        random_state=3)

print(data3X_train.shape)
print(data3X_test.shape)
print(data3Y_train.shape)
print(data3Y_test.shape)

# OVR
# for i in range(0, 3):
#     yTrain = copy.deepcopy(data3Y_train)
#     yTest = copy.deepcopy(data3Y_test)
#
#     for j in range(0, data3Y_train.shape[0]):
#         if (yTrain[j] == i):
#             yTrain[j] = 1
#         else:
#             yTrain[j] = 0
#
#     for j in range(0, data3Y_test.shape[0]):
#
#         if (yTest[j] == i):
#             yTest[j] = 1
#         else:
#             yTest[j] = 0
#
#     model3 = svm.SVC(kernel='linear')
#
#     model3.fit(data3X_train, yTrain)
#
#     # print(model2)
#
#     bias3 = model3.intercept_
#     print('intercept 3:', bias3)
#
#     print('inbuilt f1 for model3 class:', i, '-->', metrics.f1_score(yTest, model3.predict(data3X_test)))
#
#     alphas3 = model3.coef_
#
#     mt.figure(i)
#     mt.subplot(121)
#     plot_decision_regions(data3X_train, data3Y_train, model3, legend=2)
#     mt.subplot(122)
#     plot_decision_regions(data3X_test, data3Y_test, model3, legend=2)
#
#     # plot_decision_regions(data3X_train, data3Y_train, model3, legend=2)
#
# mt.show()

# OVO
for i in range(0, 3):
    for k in range(i + 1, 3):
        xTrain = copy.deepcopy(data3X_train)
        xTest = copy.deepcopy(data3X_test)
        yTrain = copy.deepcopy(data3Y_train)
        yTest = copy.deepcopy(data3Y_test)

        ovoXTrain = []
        ovoXTest = []
        ovoYTrain = []
        ovoYTest = []

        for j in range(0, data3Y_train.shape[0]):
            if (yTrain[j] == i):
                # print('i:', i)
                # ovoYTrain.append(yTrain[j])
                ovoYTrain.append(0)
                ovoXTrain.append(xTrain[j])

            elif (yTrain[j] == k):
                # print('k: ', k)
                # ovoYTrain.append(yTrain[j])
                ovoYTrain.append(1)
                ovoXTrain.append(xTrain[j])

        for j in range(0, data3Y_test.shape[0]):

            if (yTest[j] == i):
                # print('-->i:', i)
                # ovoYTest.append(yTest[j])
                ovoYTest.append(0)
                ovoXTest.append(xTest[j])
            elif (yTest[j] == k):
                # print('-->k:', k)
                # ovoYTest.append(yTest[j])
                ovoYTest.append(1)
                ovoXTest.append(xTest[j])

        ovoXTrain = np.asarray(ovoXTrain)
        ovoXTest = np.asarray(ovoXTest)
        print(ovoXTrain.shape)
        model3 = svm.SVC(kernel='linear')

        model3.fit(ovoXTrain, ovoYTrain)

        # print(model2)

        bias3 = model3.intercept_
        print('intercept 3:', bias3)

        # print(len(ovoYTest), ' -->', len(ovoXTest))

        ovoYTrain = np.asarray(ovoYTrain)
        ovoYTest = np.asarray(ovoYTest)

        print(ovoXTest.shape, ovoYTest.shape)

        pre = model3.predict(ovoXTest)
        print(ovoYTest.shape, pre.shape)
        print(ovoYTest)
        print(pre)
        f1 = f1_score(ovoYTest, pre)

        print('inbuilt f1 for model3 class:', i, ' ', k, ':', f1)

        alphas3 = model3.coef_

        mt.figure(i)
        # mt.subplot(121)
        plot_decision_regions(ovoXTrain, ovoYTrain, model3, legend=2)
        # mt.subplot(122)
        # plot_decision_regions(ovoXTest, ovoYTest, model3, legend=2)
        mt.show()

mt.show()
