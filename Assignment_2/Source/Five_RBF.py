from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import imageio
import glob
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sn

from sklearn.manifold import TSNE

x_train = []
y_train = []

data_ka = []
label_ka = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Train_val_Handwritten_Hindi_dataset/Train_val/character_1_ka/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_ka.append(lis)
    label_ka.append(0)
print(len(data_ka))

for i in range(0, 1700):
    x_train.append(data_ka[i])
    y_train.append(label_ka[i])

print(len(x_train), ' ', len(y_train))

data_kha = []
label_kha = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Train_val_Handwritten_Hindi_dataset/Train_val/character_1_kha/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_kha.append(lis)
    label_kha.append(1)

for i in range(0, 1700):
    x_train.append(data_kha[i])
    y_train.append(label_kha[i])

print(len(x_train), ' ', len(y_train))

data_ga = []
label_ga = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Train_val_Handwritten_Hindi_dataset/Train_val/character_1_ga/*.png/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_ga.append(lis)
    label_ga.append(2)

for i in range(0, 1700):
    x_train.append(data_ga[i])
    y_train.append(label_ga[i])

print(len(x_train), ' ', len(y_train))

data_gha = []
label_gha = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Train_val_Handwritten_Hindi_dataset/Train_val/character_1_gha/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_gha.append(lis)
    label_gha.append(3)

for i in range(0, 1700):
    x_train.append(data_gha[i])
    y_train.append(label_gha[i])

print(len(x_train), ' ', len(y_train))

data_kna = []
label_kna = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Train_val_Handwritten_Hindi_dataset/Train_val/character_1_kna/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_kna.append(lis)
    label_kna.append(4)

for i in range(0, 1700):
    x_train.append(data_kna[i])
    y_train.append(label_kna[i])

print(len(x_train), ' ', len(y_train))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(x_train.shape, y_train.shape)

label_ka = np.asarray(label_ka)
print(label_ka.shape)

X_scaled = preprocessing.scale(x_train)

tuned_parameters = [{'C': [10 ** -4, 10 ** -2, 10 ** 0, 10 ** 2, 10 ** 4]}]
model = GridSearchCV(svm.SVC(kernel='rbf'), tuned_parameters, cv=2, n_jobs=-1)

# model=svm.SVC(kernel='rbf')
model.fit(X_scaled, y_train)

scores = cross_val_score(model, X_scaled, y_train, cv=2, scoring='f1_macro')

print(scores)

s1 = x_train[0]
s1 = s1.reshape(1, 1024)
print(model.predict(s1))

s2 = x_train[1700]
s2 = s2.reshape(1, 1024)
print(model.predict(s2))

s3 = x_train[3400]
s3 = s3.reshape(1, 1024)
print(model.predict(s3))

s4 = x_train[5100]
s4 = s4.reshape(1, 1024)
print(model.predict(s4))

s5 = x_train[8400]
s5 = s5.reshape(1, 1024)
print(model.predict(s5))

# -------------------------------------------------------------------------
x_test = []
y_test = []

data_test_ka = []
label_test_ka = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Test_Handwritten_Hindi_dataset/Test/character_1_ka/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_test_ka.append(lis)
    label_test_ka.append(0)
print(len(data_test_ka))

for i in range(0, 300):
    x_test.append(data_test_ka[i])
    y_test.append(label_test_ka[i])

print(len(x_test), ' ', len(y_test))

data_test_kha = []
label_test_kha = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Test_Handwritten_Hindi_dataset/Test/character_1_kha/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_test_kha.append(lis)
    label_test_kha.append(1)

for i in range(0, 300):
    x_test.append(data_test_kha[i])
    y_test.append(label_test_kha[i])

print(len(x_test), ' ', len(y_test))

data_test_ga = []
label_test_ga = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Test_Handwritten_Hindi_dataset/Test/character_1_ga/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_test_ga.append(lis)
    label_test_ga.append(2)

for i in range(0, 300):
    x_test.append(data_test_ga[i])
    y_test.append(label_test_ga[i])

print(len(x_test), ' ', len(y_test))

data_test_gha = []
label_test_gha = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Test_Handwritten_Hindi_dataset/Test/character_1_gha/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_test_gha.append(lis)
    label_test_gha.append(3)

for i in range(0, 300):
    x_test.append(data_test_gha[i])
    y_test.append(label_test_gha[i])

print(len(x_test), ' ', len(y_test))

data_test_kna = []
label_test_kna = []
for im_path in glob.glob(
        "/mnt/Education/Sem1/ML/Assignments/Assignment_2/hw-2_20119/Test_Handwritten_Hindi_dataset/Test/character_1_kna/*.png"):
    im = imageio.imread(im_path)
    im = im.ravel()
    lis = list(im)
    data_test_kna.append(lis)
    label_test_kna.append(4)

for i in range(0, 300):
    x_test.append(data_test_kna[i])
    y_test.append(label_test_kna[i])

print(len(x_test), ' ', len(y_test))

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(x_test.shape, y_test.shape)

# data = np.asarray(data)
# print(data.shape)

X_test_scaled = preprocessing.scale(x_test)

c = model.predict(X_test_scaled)

f1 = classification_report(y_test, c)
print(f1)

c1 = model.predict(X_scaled)

f2 = classification_report(y_train, c1)
print(f2)

# -----------------------------------------------------------------------------------------

t = np.random.choice(8500, 2000)
rando = X_scaled[t]
randoy = y_train[t]

model_tSNE = TSNE(n_components=2, random_state=0)
tsne_data = model_tSNE.fit_transform(rando)
tsne_data = np.vstack((tsne_data.T, randoy)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.show()
