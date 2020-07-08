import os
import gzip
import numpy as np
from matplotlib import pyplot as plt
import torchvision


def load_fashion_mnist(split):
    """Loads Fashion MNIST.
    
    Args:
        split (str): train or test
        
    Returns:
        x (np.ndarray): array with images
        y (np.ndarray): array with labels
    """
    train = True if split=='train' else False
    data = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=train, transform=None, target_transform=None, download=True)

    return data.data.reshape(-1,28*28)/255., data.targets.float()

def sigmoid(t):
    """Compute sigmoid function.

    Args:
        t (np.array): argument of sigmoid function

    Returns:
        sigmoid (np.array): sigmoid function defined by `1 / (1 + exp(-t))``
    """
    return 1. / (1. + np.exp(-t))


def plot_reconstructions(x_in, x_out, out_path):
    """Plot reconstructions and show plot.

    Args:
        x_in (np.array): input images
        x_out (np.array): reconstructed images
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.figure()
    nplots = x_in.shape[0]
    for i in range(nplots):
        ax = plt.subplot(2, nplots, i+1)
        ax.imshow(x_in[i].reshape(28, 28), vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
        ax = plt.subplot(2, nplots, nplots+i+1)
        ax.imshow(x_out[i].reshape(28, 28), vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
    plt.savefig(out_path +'/recon.png')


def plot_samples(x,out_path):
    """Plot random samples and show plot.

    Args:
        x (np.array): batch of samples to be plotted
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.figure()
    nplots = len(x)
    for i in range(nplots):
        ax = plt.subplot(1, nplots, i+1)
        ax.imshow(x[i].reshape(28, 28), vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
    plt.savefig(out_path +'/samples.png')


def plot_principal_components(V, out_path):
    """Plot principal components and show plot.

    Similar to `plot_samples` but does not normalize images to have range
    between 0 and 1.

    Args:
        V (np.array): batch of first n principal components
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)    
    plt.figure()
    for i, v in enumerate(V):
        v = v.reshape(28, 28)
        ax = plt.subplot(1, len(V), 1+i)
        ax.imshow(v, cmap='gray')
        ax.axis('off')
    plt.savefig(out_path+'/pc.png')


def plot_cummulative_distribution(p_cum, out_path):
    """Plot commulative distribution of singluar values.

    Args:
        p_cum (np.array): cummulative distribution of principal values
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.figure()
    plt.plot(p_cum)
    plt.xlabel('# principal components')
    plt.ylabel('fraction of explained variance')
    plt.savefig(out_path+'/cum_dist.png')
