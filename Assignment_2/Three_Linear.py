import copy
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mt
import numpy as np
from sklearn import svm
from sklearn import metrics
import h5py
from sklearn.metrics import f1_score, accuracy_score
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
#
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
print('Accuracy for test:', accuracy_score(data1Y_test, out1))

TN34 = FP34 = FN34 = TP34 = 0
som = copy.deepcopy(data1Y_test)
som = np.asarray(som)

for i in range(20):
    #     print(predict5[i],"==",wow[i,0])
    if (out1[i] == som[i] and out1[i] == 0):
        TN34 = TN34 + 1
    if (out1[i] == som[i] and out1[i] == 1):
        TP34 = TP34 + 1
    if (out1[i] != som[i] and out1[i] == 1 and som[i] == 0):
        FP34 = FP34 + 1
    if (out1[i] != som[i] and out1[i] == 0 and som[i] == 1):
        FN34 = FN34 + 1

confusion1 = np.array([[TN34, FP34], [FN34, TP34]])
print('confusion1:')
print(confusion1)

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

# mt.subplot(221)
mt.figure('data_1 Linear')
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias1, 0, bias1])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xlabel('x1')
mt.ylabel('x2')
mt.title('data_1 Linear Kernel')
mt.axis('tight')
mt.show()

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

print('Accuracy for test:', accuracy_score(data2Y_test, out2))

TN34 = FP34 = FN34 = TP34 = 0
som = copy.deepcopy(data2Y_test)
som = np.asarray(som)

for i in range(20):
    if (out2[i] == som[i] and out2[i] == 0):
        TN34 = TN34 + 1
    if (out2[i] == som[i] and out2[i] == 1):
        TP34 = TP34 + 1
    if (out2[i] != som[i] and out2[i] == 1 and som[i] == 0):
        FP34 = FP34 + 1
    if (out2[i] != som[i] and out2[i] == 0 and som[i] == 1):
        FN34 = FN34 + 1

confusion2 = np.array([[TN34, FP34], [FN34, TP34]])
print('confusion2:')
print(confusion2)

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

# mt.subplot(222)
mt.figure('data_2 Linear')
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias2, 0, bias2])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xlabel('x1')
mt.ylabel('x2')
mt.title('data_2 Linear Kernel')
mt.axis('tight')
mt.show()

print('-------------------------------------------')

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
for i in range(0, 3):
    yTrain = copy.deepcopy(data3Y_train)
    yTest = copy.deepcopy(data3Y_test)

    for j in range(0, data3Y_train.shape[0]):
        if (yTrain[j] == i):
            yTrain[j] = 1
        else:
            yTrain[j] = 0

    for j in range(0, data3Y_test.shape[0]):

        if (yTest[j] == i):
            yTest[j] = 1
        else:
            yTest[j] = 0

    model3 = svm.SVC(kernel='linear')

    model3.fit(data3X_train, yTrain)

    # print(model3)

    bias3 = model3.intercept_
    print('intercept 3:', bias3)

    print('inbuilt f1 for model3 class:', i, '-->', f1_score(yTest, model3.predict(data3X_test)))

    bias3 = model3.intercept_
    print('intercept 3:', bias3)
    alphas3 = model3.coef_

    out3 = []
    for i in range(0, len(data3X_test)):
        cls = predict(alphas3[0], data3X_test[i], bias3)  # my predict function
        if (cls >= 0):
            out3.append(1)
        else:
            out3.append(0)

    print('my f1 for model3:', f1_score(yTest, out3))

    print('Accuracy for test:', accuracy_score(yTest, out3))

    TN34 = FP34 = FN34 = TP34 = 0
    som = copy.deepcopy(yTest)
    som = np.asarray(som)

    for i in range(20):
        if (out3[i] == som[i] and out3[i] == 0):
            TN34 = TN34 + 1
        if (out3[i] == som[i] and out3[i] == 1):
            TP34 = TP34 + 1
        if (out3[i] != som[i] and out3[i] == 1 and som[i] == 0):
            FP34 = FP34 + 1
        if (out3[i] != som[i] and out3[i] == 0 and som[i] == 1):
            FN34 = FN34 + 1

    confusion3 = np.array([[TN34, FP34], [FN34, TP34]])
    print('confusion3:')
    print(confusion3)

    alphas3 = model3.coef_

    x_min = -20
    x_max = 20
    y_min = -20
    y_max = 20

    X = data3X_train
    Y = data3Y_train

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    inp = np.c_[XX.ravel(), YY.ravel()]
    # print('-->', inp.shape)

    Z = []
    for k in range(0, len(inp)):
        query = inp[k]
        c = predict(alphas3[0], query, bias3)  # my predict function
        Z.append(c)

    # Z = model3.decision_function(inp)

    Z = np.array(Z)
    Z = Z.reshape(XX.shape)
    print('bias3:', bias3)

    name = 'data_3 OVR'
    mt.figure(name)
    # mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
    # mt.contour(XX, YY, Z >= 0, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias3, 0, bias3])
    #
    mt.scatter(X[:, 0], X[:, 1], c=Y)
    #
    # mt.xlim(x_min, x_max)
    # mt.ylim(y_min, y_max)
    mt.xlabel('x1')
    mt.ylabel('x2')
    mt.title('data_3 Linear Kernel')
    # mt.axis('tight')

    # plot_decision_regions(data3X_train, data3Y_train, model3, legend=2)
    # plot_decision_regions(data3X_test, data3Y_test, model3, legend=2)

    plot_decision_regions(data3X_train, data3Y_train, model3, legend=2)
    # mt.show()

mt.show()

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

        print(set(ovoYTrain))
        print(set(ovoYTest))

        ovoYTrain = np.asarray(ovoYTrain)
        ovoYTest = np.asarray(ovoYTest)

        print(ovoXTest.shape, ovoYTest.shape)

        pre = model3.predict(ovoXTest)
        print(ovoYTest.shape, pre.shape)
        print(ovoYTest)
        print(pre)
        f1 = f1_score(ovoYTest, pre)

        print('inbuilt f1 for model3 class:', i, ' ', k, ':', f1)

        bias3 = model3.intercept_
        print('intercept 3:', bias3)
        alphas3 = model3.coef_



        mt.figure('data_3 OVO Train')
        mt.title('data_3 Linear Kernel Train')
        mt.xlabel('x1')
        mt.ylabel('x2')
        plot_decision_regions(ovoXTrain, ovoYTrain, model3, legend=2)

        mt.figure('data_3 OVO Test')
        mt.title('data_3 Linear Kernel Test')
        mt.xlabel('x1')
        mt.ylabel('x2')
        plot_decision_regions(ovoXTest, ovoYTest, model3, legend=2)

        mt.show()

print('-------------------------------------------')

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

print('Accuracy for test:', accuracy_score(data4Y_test, out4))

TN34 = FP34 = FN34 = TP34 = 0
som = copy.deepcopy(data4Y_test)
som = np.asarray(som)

for i in range(20):
    if (out4[i] == som[i] and out4[i] == 0):
        TN34 = TN34 + 1
    if (out4[i] == som[i] and out4[i] == 1):
        TP34 = TP34 + 1
    if (out4[i] != som[i] and out4[i] == 1 and som[i] == 0):
        FP34 = FP34 + 1
    if (out4[i] != som[i] and out4[i] == 0 and som[i] == 1):
        FN34 = FN34 + 1

confusion4 = np.array([[TN34, FP34], [FN34, TP34]])
print('confusion4:')
print(confusion4)

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

# mt.subplot(224)
mt.figure('data_4 Linear')
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias4, 0, bias4])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xlabel('x1')
mt.ylabel('x2')
mt.title('data_4 Linear Kernel')
mt.axis('tight')
mt.show()

print('-------------------------------------------')

mt.show()
