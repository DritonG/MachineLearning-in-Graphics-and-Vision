import os
import gzip
import numpy as np
from sklearn.neighbors import KDTree

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


train_img, train_label = load_mnist('/path/to/fashion_mnist', kind='train', each=10)
test_img, test_label = load_mnist('/path/to/fashion_mnist', kind='t10k', each=10)

# TODO
