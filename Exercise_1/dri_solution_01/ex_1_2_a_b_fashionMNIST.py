# Ravlija xxxxxx
# Goxhufi 4233242 

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import os
import gzip


def load_mnist(path, kind = 'train', each=1):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz'% kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    images = images[::each,:]
    labels = labels[::each]

    return images, labels

train_img, train_label = load_mnist(path='C:/Users/fs/Datasets/fashion-mnist-master/data/fashion/', kind='train', each=10)
test_img, test_label = load_mnist(path='C:/Users/fs/Datasets/fashion-mnist-master/data/fashion/', kind='t10k', each=10)


# task 1_2_a

tree = KDTree(train_img)

def calculate_top_k_acc(k):
    dist, ind = tree.query(test_img, k)

    #print('ind.shape: ', ind.shape)
    #ind[n,:] contains the indeces of the k nearest neighbors of test_img[n]
    #the true label of test_img[n] is test_label[n]

    #is the correct label amongst the k nearest neighbors?:
    label_abundance = [0] * len(test_img)
    for i in range(len(test_img)):
        true_label = test_label[i]
        for j in range(k):
            neighbor_label = train_label[ind[i,j]]
            if neighbor_label == true_label:
                label_abundance[i] = 1

    top_k_acc = np.mean(label_abundance)
    print('k:', k, 'top_k_acc: ', top_k_acc)
    return k, top_k_acc


k_list = []
top_k_acc_list = []
for i in range(1,11):
    k, top_k_acc = calculate_top_k_acc(i)
    k_list.append(k)
    top_k_acc_list.append(top_k_acc)


# plt.plot(k_list, top_k_acc_list)
# plt.xlabel('# nearest neighbors')
# plt.ylabel('top k accuracy')
# plt.show()


# task 1_2_b

# Consider the 1 NN-classifier for two classes 'Pullover' (2) and "shirt" (6)
# ?-> so only use data with label 2 and 6
ind_pullover_train = np.where(train_label == 2)[0]
ind_pullover_test = np.where(test_label == 2)[0]

ind_train_shirt = np.where(train_label == 6)[0]
ind_test_shirt = np.where(test_label == 6)[0]

# binary training data
# shorter way?!
x_train_pullover = train_img[ind_pullover_train]
y_train_pullover = np.ones(len(x_train_pullover))
x_train_shirt = train_img[ind_train_shirt]
y_train_shirt = np.zeros(len(x_train_shirt))

x_test_pullover = test_img[ind_pullover_test]
y_test_pullover = np.ones(len(x_test_pullover))
x_test_shirt = test_img[ind_test_shirt]
y_test_shirt = np.zeros(len(x_test_shirt))

x_train = np.vstack((x_train_pullover, x_train_shirt))
y_train = np.hstack((y_train_pullover, y_train_shirt))

x_test = np.vstack((x_test_pullover, x_test_shirt))
y_test = np.hstack((y_test_pullover, y_test_shirt))

# print('x_train.shape', x_train.shape)
# print('y_train.shape', y_train.shape)
# print('x_test.shape', x_test.shape)
# print('y_test.shape', y_test.shape)

binary_class = KDTree(x_train)
dist_b, ind_b = binary_class.query(x_test, k=1)

predicted_labels = y_train[ind_b]
predicted_labels = predicted_labels.reshape(len(predicted_labels,))
actual_labels = y_test

# print('predicted_labels.shape', predicted_labels.shape)
# print('actual_labels.shape', actual_labels.shape)

# count number of true positives T_p, false positives F_p and false negatives F_n
T_p = 0
F_p = 0
F_n = 0
for i in range(len(predicted_labels)):
    if predicted_labels[i] == 1 and actual_labels[i] == 1:
        T_p += 1
    if predicted_labels[i] == 1 and actual_labels[i] == 0:
        F_p += 1
    if predicted_labels[i] == 0 and actual_labels[i] == 1:
        F_n += 1

precision = T_p / (T_p + F_p)
recall = T_p / (T_p + F_n)
print('precision', precision)
print('recall',  recall)
