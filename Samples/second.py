

from mnist import MNIST
# mndata = MNIST('home/structubhani007/Desktop/ML Assignment/mnistData/')


mndata= MNIST('/home/subhani007/Desktop/ML Assignment/extracted')
mndata.gz = False
images_train, labels_train = mndata.load_training()


images_test, labels_test = mndata.load_testing()
print(len(images_train))
print(len(labels_train))
