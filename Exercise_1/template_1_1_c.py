import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


def kdtree_search(tree, queries):
    # TODO
    pass


if __name__ == '__main__':
    # timings for different D
    times = []

    # kd-tree search
    for D in range(1, 500, 10):
        # TODO
        pass

    # plot to file
    plt.plot(range(1, 500, 10), times)
    plt.title('Query Times')
    plt.xlabel('dimension (D)')
    plt.ylabel('time (ms)')
    plt.savefig('1_1_c.png', bbox_inches='tight')