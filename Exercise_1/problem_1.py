import numpy as np
import matplotlib.pyplot as plt

import timeit

# with the seed for initialization
rng = np.random.default_rng(1)
vals = rng.random(10)



# 1. a)
# The complexity of this method is O(d*n*m) where d is the dimension 
# of feature vector, n is the number of examples and m is the number
# of query vectors.
# 
# 1. b)
# D = 128
# N = 20,000
# There is A single vector has to be compared with 
# 	30 FPS * 120 s = 3,600 frames = 3,600 * N vectors = 7.2 * 10^7 vectors
# vectors in one video, and each one of these has to be compared with
# all of the vectors from another video (i.e. also 7.2 * 10^7 vectors).

def exhaustive_search(dataset, q):
	'''
	Finds the nearest neighbor of q in the given dataset.
	:param dataset: matrix with examples in rows
	:param q: target point

	:returns the index of nearest neighbor vector of q
		in the given dataset
	'''
	distances = np.linalg.norm(dataset - q, axis=1)
	return np.argmin(distances) 

times = []
dimensions = [x for x in range(1, 492, 10)]
for d in dimensions:
	dataset = rng.random(d*(2**10))
	dataset = dataset.reshape(-1,d)
		
	# benchmark the query time
	start = timeit.default_timer()

	for i in range(len(dataset)):
		q = dataset[i]
		dataset_no_q = np.delete(dataset, i, axis=0)
		# Could account for changed indices due to deletion
		exhaustive_search(dataset_no_q, q)

	end = timeit.default_timer()
	times.append(end - start)
	# print(d)
	# print(times)
	print(np.sum(times))
plt.plot(dimensions, times)
plt.xlabel('dimension')
plt.ylabel('query time')
plt.savefig('./task1a.png')
	




	
