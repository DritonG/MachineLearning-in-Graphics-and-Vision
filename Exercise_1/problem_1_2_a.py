'''
group member 1: Driton Goxhufi      xxxxxxx     driton.goxhufi@student.uni-tuebingen.de
group member 2: Damir Ravlija       5503184     damir.ravlija@student.uni-tuebingen.de
'''


import os
import gzip
import numpy as np
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt

import problem_1_1_c

"""
see: https://github.com/zalandoresearch/fashion-mnist
"""

def load_mnist(path, kind='train', each=1):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images[::each, :], labels[::each]


if __name__ == '__main__':

    print('Loading mnist', end='\r')

    # 'fashion_mnist' should be in the working directory
    train_img, train_label = load_mnist('fashion_mnist', kind='train', each=10)
    test_img, test_label = load_mnist('fashion_mnist', kind='t10k', each=10)

    # Task 1.2.a)
    largest_K = 10
    accuracies = []
    for k in range(1, largest_K):
        tree = KDTree(train_img)
        nearest_nbrs = tree.query(test_img, k, return_distance=False)
        correct = 0
        for i in range(len(nearest_nbrs)):
            correct += test_label[i] in train_label[nearest_nbrs[i]]

        accuracy = correct / len(test_label)
        accuracies.append(accuracy)
        print(f'Generating results for 2.a): {k:2d}/{largest_K:2d}\tAccuracy: {accuracy:.3f}', end='\r')
        

    print(60 * '-')
    print('Done. Results for 2.a) generated.')

    plt.plot(list(range(1, largest_K)), accuracies)
    plt.title('Top-K accuracy')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.savefig('1_2_a.png', bbox_inches='tight')
