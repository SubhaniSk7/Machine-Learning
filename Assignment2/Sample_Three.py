import copy
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mt
import numpy as np
from sklearn import svm
import h5py
from sklearn.metrics import f1_score
from mlxtend.plotting import plot_decision_regions

data1 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_1.h5', 'r+')
data2 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_2.h5', 'r+')
data3 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_3.h5', 'r+')
data4 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_4.h5', 'r+')


# ------------------------------------------------------------------
def predict(alphas, query, b):
    prod = np.dot(alphas, query.T)
    s = prod + b
    # if (s >= 0):
    #     return 1
    # else:
    #     return 0
    return s


def predict_3(x):
    pass


def predict_4(alphas, query, b):
    prod = np.dot(alphas, query.T)
    s = prod + b
    return s


# ------------------------------------------------------------------

# print(list(data1.keys()))
data1_X = data1['x']
data1_Y = data1['y']

data1X_Values = np.array(data1_X.value)
data1Y_Values = np.array(data1_Y.value)
print(data1X_Values.shape)
print(data1Y_Values.shape)

data1X_train, data1X_test, data1Y_train, data1Y_test = train_test_split(data1X_Values, data1Y_Values, test_size=0.2,
                                                                        random_state=20)

model1 = svm.SVC(kernel='linear')

model1.fit(data1X_train, data1Y_train)

# print(model1)

print('inbuilt f1 for model1:', f1_score(data1Y_test, model1.predict(data1X_test)))

bias1 = model1.intercept_
print('intercept:', bias1)
alphas1 = model1.coef_

out1 = []
for i in range(0, len(data1X_test)):
    cls = predict(alphas1[0], data1X_test[i], bias1)  # my predict function
    if (cls >= 0):
        out1.append(1)
    else:
        out1.append(0)
    # out1.append(cls)
print('my f1 for model1:', f1_score(data1Y_test, out1))

x_min = -3
x_max = 3
y_min = -3
y_max = 3

X = data1X_train
Y = data1Y_train

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

inp = np.c_[XX.ravel(), YY.ravel()]
# print('-->', inp.shape)

Z = []
for i in range(0, len(inp)):
    query = inp[i]
    c = predict(alphas1[0], query, bias1)  # my predict function
    Z.append(c)

# Z = model1.decision_function(inp)

Z = np.array(Z)
Z = Z.reshape(XX.shape)
print(XX.shape, YY.shape, Z.shape)
print('bias1:', bias1)

mt.subplot(221)
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias1, 0, bias1])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xticks(())
mt.yticks(())
mt.axis('tight')

print('-------------------------------------------')

# print(list(data2.keys()))
data2_X = data2['x']
data2_Y = data2['y']

data2X_Values = np.array(data2_X.value)
data2Y_Values = np.array(data2_Y.value)

print(data2X_Values.shape)
print(data2Y_Values.shape)

data2X_train, data2X_test, data2Y_train, data2Y_test = train_test_split(data2X_Values, data2Y_Values, test_size=0.2,
                                                                        random_state=3)

model2 = svm.SVC(kernel='linear')

model2.fit(data2X_train, data2Y_train)

# print(model2)

print('inbuilt f1 for model2:', f1_score(data2Y_test, model2.predict(data2X_test)))

bias2 = model2.intercept_
print('intercept 2:', bias2)

alphas2 = model2.coef_
# print(alphas2)

out2 = []
for i in range(0, len(data2X_test)):
    cls = predict(alphas2[0], data2X_test[i], bias2)  # my predict function
    if (cls >= 0):
        out2.append(1)
    else:
        out2.append(0)
    # out2.append(cls)

print('my f1 for model2:', f1_score(data2Y_test, out2))

x_min = -3
x_max = 3
y_min = -3
y_max = 3

X = data2X_train
Y = data2Y_train

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

inp = np.c_[XX.ravel(), YY.ravel()]
print('-->', inp.shape)

Z = []

for i in range(0, len(inp)):
    query = inp[i]
    c = predict(alphas2[0], query, bias2)  # my predict function
    Z.append(c)

# Z = model2.decision_function(inp)

Z = np.array(Z)
Z = Z.reshape(XX.shape)
print(XX.shape, YY.shape, Z.shape)
print('bias2:', bias2)

mt.subplot(222)
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias2, 0, bias2])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xticks(())
mt.yticks(())
mt.axis('tight')

print('-------------------------------------------')

# #
# # print(list(data3.keys()))
# data3_X = data3['x']
# data3_Y = data3['y']
#
# data3X_Values = np.array(data3_X.value)
# data3Y_Values = np.array(data3_Y.value)
#
# print(data3X_Values.shape)
# print(data3Y_Values.shape)
# data3X_train, data3X_test, data3Y_train, data3Y_test = train_test_split(data3X_Values, data3Y_Values, test_size=0.2,
#                                                                         random_state=3)
#
# model3 = svm.SVC(kernel='linear')
#
# model3.fit(data3X_train, data3Y_train)
#
# # print(model3)
#
# # print('intercept 3:', model3.intercept_)
# print('f1 for model3:', f1_score(data3Y_test, model3.predict(data3X_test)))
# #
# print('-------------------------------------------')


# print(list(data4.keys()))
data4_X = data4['x']
data4_Y = data4['y']

data4X_Values = np.array(data4_X.value)
data4Y_Values = np.array(data4_Y.value)

print(data4X_Values.shape)
print(data4Y_Values.shape)

data4X_train, data4X_test, data4Y_train, data4Y_test = train_test_split(data4X_Values, data4Y_Values, test_size=0.2,
                                                                        random_state=3)

model4 = svm.SVC(kernel='linear')

model4.fit(data4X_train, data4Y_train)

# print(model4)

print('inbuilt f1 for model4:', f1_score(data4Y_test, model4.predict(data4X_test)))

bias4 = model4.intercept_
print('intercept 4:', bias4)
alphas4 = model4.coef_

out4 = []
for i in range(0, len(data4X_test)):
    cls = predict(alphas4[0], data4X_test[i], bias4)  # my predict function
    if (cls >= 0):
        out4.append(1)
    else:
        out4.append(0)
    # out4.append(cls)

print('my f1 for model2:', f1_score(data4Y_test, out4))

x_min = -3
x_max = 3
y_min = -3
y_max = 3

X = data4X_train
Y = data4Y_train

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

inp = np.c_[XX.ravel(), YY.ravel()]
print('-->', inp.shape)

Z = []
for i in range(0, len(inp)):
    query = inp[i]
    c = predict_4(alphas4[0], query, bias4)  # my predict function
    Z.append(c)

# Z = model4.decision_function(inp)

Z = np.array(Z)
Z = Z.reshape(XX.shape)
print(XX.shape, YY.shape, Z.shape)
print('bias4:', bias4)

mt.subplot(224)
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias4, 0, bias4])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xticks(())
mt.yticks(())
mt.axis('tight')

print('-------------------------------------------')

mt.show()
