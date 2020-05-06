# Ravlija xxxxxx
# Goxhufi 4233242

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# dataset vectors
N = 2**10


def kdtree_search(tree, queries):
    dist, ind = tree.query(queries, k=1)
    return dist, ind



if __name__ == '__main__':
    # timings for different D
    times = []

    # kd-tree search
    for D in range(1, 500, 10):
        X = np.random.random((N,D))

        tree = KDTree(X)
        queries = X

        query_start = time.clock()
        dist, ind = kdtree_search(tree, queries)
        query_end = time.clock()

        query_time = query_end - query_start
        times.append(query_time)


    # plot to file
    plt.plot(range(1, 500, 10), times)
    plt.title('Query Times')
    plt.xlabel('dimension (D)')
    plt.ylabel('time (ms)')
    plt.savefig('1_1_c.png', bbox_inches='tight')
    plt.show()
