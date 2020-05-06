# Ravlija xxxxxx
# Goxhufi 4233242

import numpy as np
import time
import matplotlib.pyplot as plt


# ex 1.1
# Dataset X containing N vectors x element R**D
# find nearest neighbor NN for query q

# a)
# randomly generate N=2**10 points
N = 2**10

# exhaustive search: try all possibilities
def exhaustive_search(D):
    # can use np array X with each row representing one vector x
    X = np.random.rand(N, D)

    # 'take all elements from X as a query'
    q = X
    #its nearest neighbor will be the point itself???


    #placeholder for smallest dist, smallest dist index, query index
    smallest_dist = [0] * N
    smallest_dist_index = [0] * N
    query_index = range(N)

    query_start = time.perf_counter()

    for i in range(N):

        for j in range(N):
            # calculate distance between each x and query q
            eucl_dist = np.linalg.norm(X[j]-q[i])
            if eucl_dist < smallest_dist[i]:
                smallest_dist[i] = eucl_dist
                smallest_dist_index[i] = j

    query_end = time.perf_counter()
    query_time = query_end - query_start

    return D, query_time, smallest_dist_index, smallest_dist

def benchmark_exhaustive_search():
    vector_dimensions = []
    computation_time = []
    for i in range(1, 492, 10):
        D, query_time, smallest_dist_index, smallest_dist = exhaustive_search(i)
        vector_dimensions.append(i)
        computation_time.append(query_time)

    plt.plot(vector_dimensions, computation_time)
    plt.xlabel('vector dimensions')
    plt.ylabel('computation time [s]')
    plt.show()

benchmark_exhaustive_search()


# qq what is the complexity of this method???


# ex 1.1 b)
#dimension of vectors
D = 128
#number of vectors to check in 2 minute video:
nov = 20000 * 30 * 60 * 2

#check for N = 100
N = 100

D, query_time, smallest_dist_index, smallest_dist = exhaustive_search(D)

#assume linear dependency
HD_computation_time = query_time * nov/N
print('HD_computation_time [s]: {0:.2f}'.format(HD_computation_time))
print('HD_computation_time [hours]: {0:.2f}'.format(HD_computation_time/3600))
