import copy
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mt
import numpy as np
from sklearn import svm
import h5py
from sklearn.metrics import f1_score, accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics.pairwise import rbf_kernel

data1 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_1.h5', 'r+')
data2 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_2.h5', 'r+')
data3 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_3.h5', 'r+')
data4 = h5py.File('/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/data_4.h5', 'r+')


# ------------------------------------------------------------------

def predict(coeff, inner, b):
    ans = sum(coeff * inner) + b

    return ans


def predict_3(x):
    pass


def predict_4(alphas, query, b):
    prod = np.dot(alphas, query.T)
    s = prod + b
    return s


# ------------------------------------------------------------------

print(list(data1.keys()))
data1_X = data1['x']
data1_Y = data1['y']

data1X_Values = np.array(data1_X.value)
data1Y_Values = np.array(data1_Y.value)
print(data1X_Values.shape)
print(data1Y_Values.shape)

data1X_train, data1X_test, data1Y_train, data1Y_test = train_test_split(data1X_Values, data1Y_Values, test_size=0.2,
                                                                        random_state=20)

model1 = svm.SVC(kernel='rbf')

model1.fit(data1X_train, data1Y_train)

# print(model1)

print('inbuilt f1 for model1:', f1_score(data1Y_test, model1.predict(data1X_test)))

bias1 = model1.intercept_
print('intercept:', bias1)
svIndex = model1.support_
# print(svIndex)

alphas1 = model1.dual_coef_

alp = [0] * data1X_train.shape[0]
for i in range(0, len(svIndex)):
    alp[svIndex[i]] = alphas1[0][i]
# print(alp)

alphas1 = alp
alphas1 = np.asarray(alphas1)
sm1 = rbf_kernel(data1X_train, data1X_test, gamma=0.7)

print(sm1.shape)
out1 = []
for i in range(0, len(data1X_test)):
    inner = sm1[:, i]
    cls = predict(alphas1, inner, bias1)  # my predict function
    if (cls >= 0):
        out1.append(1)
    else:
        out1.append(0)

print('my f1 for model1:', f1_score(data1Y_test, out1))

print('Accuracy for test:', accuracy_score(data1Y_test, out1))

TN34 = FP34 = FN34 = TP34 = 0
som = copy.deepcopy(data1Y_test)
som = np.asarray(som)

for i in range(20):
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

sm1 = rbf_kernel(data1X_train, inp, gamma=0.7)

Z = []
for i in range(0, len(inp)):
    inner = sm1[:, i]
    cls = predict(alphas1, inner, bias1)  # my predict function
    Z.append(cls)

# Z = model1.decision_function(inp)

Z = np.asarray(Z)
Z = Z.reshape(XX.shape)

print(XX.shape, YY.shape, Z.shape)
print('bias1:', bias1)

mt.figure('data_1_RBF')
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z > 0, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias1, 0, bias1])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xlabel('x1')
mt.ylabel('x2')
mt.title('data_1 RBF Kernel')
mt.axis('tight')
mt.show()
print('-------------------------------------------')

print(list(data2.keys()))
data2_X = data2['x']
data2_Y = data2['y']

data2X_Values = np.array(data2_X.value)
data2Y_Values = np.array(data2_Y.value)

print(data2X_Values.shape)
print(data2Y_Values.shape)

data2X_train, data2X_test, data2Y_train, data2Y_test = train_test_split(data2X_Values, data2Y_Values, test_size=0.2,
                                                                        random_state=3)

model2 = svm.SVC(kernel='rbf')

model2.fit(data2X_train, data2Y_train)

# print(model2)

print('inbuilt f1 for model2:', f1_score(data2Y_test, model2.predict(data2X_test)))

bias2 = model2.intercept_
print('intercept 2:', bias2)
svIndex = model2.support_
# print(svIndex)

alphas2 = model2.dual_coef_

alp = [0] * data2X_train.shape[0]
for i in range(0, len(svIndex)):
    alp[svIndex[i]] = alphas2[0][i]

alphas2 = alp
alphas2 = np.asarray(alphas2)
sm2 = rbf_kernel(data2X_train, data2X_test, gamma=0.7)

print(sm2.shape)
out2 = []
for i in range(0, len(data2X_test)):
    inner = sm2[:, i]
    cls = predict(alphas2, inner, bias2)  # my predict function
    if (cls >= 0):
        out2.append(1)
    else:
        out2.append(0)

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

sm2 = rbf_kernel(data2X_train, inp, gamma=0.7)

Z = []
for i in range(0, len(inp)):
    inner = sm2[:, i]
    cls = predict(alphas2, inner, bias2)  # my predict function
    Z.append(cls)

# Z = model2.decision_function(inp)

Z = np.asarray(Z)
Z = Z.reshape(XX.shape)

print(XX.shape, YY.shape, Z.shape)
print('bias2:', bias2)

mt.figure('data_2 RBF')
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z > 0, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias2, 0, bias2])

mt.scatter(X[:, 0], X[:, 1], c=Y)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xlabel('x1')
mt.ylabel('x2')
mt.title('data_2 RBF Kernel')
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

    model3 = svm.SVC(kernel='rbf')

    model3.fit(data3X_train, yTrain)

    # print(model3)

    bias3 = model3.intercept_
    print('intercept 3:', bias3)

    print('inbuilt f1 for model3 class:', i, '-->', f1_score(yTest, model3.predict(data3X_test)))

    bias3 = model3.intercept_
    print('intercept 3:', bias3)
    alphas3 = model3.dual_coef_

    out3 = []

    alphas3 = model3.dual_coef_

    x_min = -20
    x_max = 20
    y_min = -20
    y_max = 20

    X = data3X_train
    Y = data3Y_train

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    inp = np.c_[XX.ravel(), YY.ravel()]

    Z = model3.decision_function(inp)

    Z = np.array(Z)
    Z = Z.reshape(XX.shape)
    print('bias3:', bias3)

    # mt.figure('data_3 OVR Train')
    # mt.title('data_3 OVR RBF Train')
    # mt.xlabel('x1')
    # mt.ylabel('x2')
    # plot_decision_regions(data3X_train, data3Y_train, model3, legend=2)
    #
    # mt.figure('data_3 OVR Test')
    # mt.title('data_3 OVR RBF Test')
    # mt.xlabel('x1')
    # mt.ylabel('x2')
    # plot_decision_regions(data3X_test, data3Y_test, model3, legend=2)

    mt.figure('data_3 OVR RBF')
    mt.title('data_3 OVR RBF')
    mt.xlabel('x1')
    mt.ylabel('x2')
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
                ovoYTrain.append(0)
                ovoXTrain.append(xTrain[j])

            elif (yTrain[j] == k):
                ovoYTrain.append(1)
                ovoXTrain.append(xTrain[j])

        for j in range(0, data3Y_test.shape[0]):

            if (yTest[j] == i):
                ovoYTest.append(0)
                ovoXTest.append(xTest[j])
            elif (yTest[j] == k):
                ovoYTest.append(1)
                ovoXTest.append(xTest[j])

        ovoXTrain = np.asarray(ovoXTrain)
        ovoXTest = np.asarray(ovoXTest)
        print(ovoXTrain.shape)
        model3 = svm.SVC(kernel='rbf')

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
        alphas3 = model3.dual_coef_

        mt.figure('data_3 OVO Train')
        mt.title('data_3 Linear Kernel Train')
        mt.xlabel('x1')
        mt.ylabel('x2')
        plot_decision_regions(ovoXTrain, ovoYTrain, model3, legend=2)

        mt.figure('data_3 OVO Test')
        mt.title('data_3 RBF Kernel Test')
        mt.xlabel('x1')
        mt.ylabel('x2')
        plot_decision_regions(ovoXTest, ovoYTest, model3, legend=2)

        mt.show()

print('-------------------------------------------')

print(list(data4.keys()))
data4_X = data4['x']
data4_Y = data4['y']

data4X_Values = np.array(data4_X.value)
data4Y_Values = np.array(data4_Y.value)

print(data4X_Values.shape)
print(data4Y_Values.shape)

data4X_train, data4X_test, data4Y_train, data4Y_test = train_test_split(data4X_Values, data4Y_Values, test_size=0.2,
                                                                        random_state=3)

model4 = svm.SVC(kernel='rbf')

model4.fit(data4X_train, data4Y_train)

# print(model4)

print('inbuilt f1 for model4:', f1_score(data4Y_test, model4.predict(data4X_test)))

bias4 = model4.intercept_
print('intercept 4:', bias4)
svIndex = model4.support_
# print(svIndex)

alphas4 = model4.dual_coef_

alp = [0] * data4X_train.shape[0]
for i in range(0, len(svIndex)):
    alp[svIndex[i]] = alphas4[0][i]

alphas4 = alp
alphas4 = np.asarray(alphas4)
sm4 = rbf_kernel(data4X_train, data4X_test, gamma=0.7)

print(sm4.shape)
out4 = []
for i in range(0, len(data4X_test)):
    inner = sm4[:, i]
    cls = predict(alphas4, inner, bias4)  # my predict function
    if (cls >= 0):
        out4.append(1)
    else:
        out4.append(0)

print('my f1 for model4:', f1_score(data4Y_test, out4))

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

sm4 = rbf_kernel(data4X_train, inp, gamma=0.7)

Z = []
for i in range(0, len(inp)):
    inner = sm4[:, i]
    cls = predict(alphas4, inner, bias4)  # my predict function
    Z.append(cls)

# Z = model4.decision_function(inp)

Z = np.asarray(Z)
Z = Z.reshape(XX.shape)

print(XX.shape, YY.shape, Z.shape)
print('bias4:', bias4)

mt.figure('data_4 RBF')
mt.pcolormesh(XX, YY, Z > 0, cmap=mt.cm.Paired)
mt.contour(XX, YY, Z > 0, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-bias4, 0, bias4])

mt.scatter(X[:, 0], X[:, 1], c=Y, s=1)

mt.xlim(x_min, x_max)
mt.ylim(y_min, y_max)
mt.xlabel('x1')
mt.ylabel('x2')
mt.title('data_4 RBF Kernel')
mt.axis('tight')
mt.show()

print('-------------------------------------------')

mt.show()
