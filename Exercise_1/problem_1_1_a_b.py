'''
group member 1: Driton Goxhufi    	xxxxxxx     driton.goxhufi@student.uni-tuebingen.de
group member 2: Damir Ravlija       5503184     damir.ravlija@student.uni-tuebingen.de
'''

import numpy as np
import matplotlib.pyplot as plt

import timeit

# with the seed for initialization
rng = np.random.default_rng(1)


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


def create_dataset(N, D):
	'''
	Helper function.
	Creates a dataset with N number of D-dimensional feature vectors.
	Vector coefficients are drawn from a uniform distribution.
	:param N: number of vectors
	:param D: dimension of vectors

	:returns a matrix of shape (N x D) that contains feature vectors in rows
	'''
	dataset = rng.random((N, D))
	return dataset

	

def time_exhaustive_search(N, D):
	'''
	Helper function.
	Benchmarks the query time of exhaustive search for a nearest neighbor.
	Repeatedly queries the exhaustive search with all N D-dimensional vectors
	in the dataset.
	:param D: dimension of vectors
	:param N: number of vectors
	
	:returns time to query exhaustive search with all elements in the dataset
	'''
	dataset = create_dataset(N, D)

	start = timeit.default_timer()

	for i in range(len(dataset)):
		q = dataset[i]
		exhaustive_search(dataset, q)

	end = timeit.default_timer()

	return end - start


# Task 1.1.a)
# The complexity of this method is O(d*(n^2)) where d is the dimension 
# of feature vector and n is the number of examples.
def results_a(dimensions):
	'''
	Produces query times of exhaustive search over all elements in a dataset for
	datasets with 1024 vectors with different 'dimensions'.
	:param dimensions: array of 'dimensions' whose query times should be calculated

	:returns query times corresponding to the given 'dimensions'
	'''
	times = []
	for d in dimensions:
		time = time_exhaustive_search(1024, d)
		times.append(time)
		print(f'Generating results for 1.a): {d:3d}/{dimensions[-1]:3d}\tTime: {time:4.2f}', end='\r')
	print(60 * '-')
	print('Done. Results for 1.a) generated.')

	return times


# Task 1.1.b)
# D = 128
# N = 20,000
# A single vector has to be compared with 
# 	30 FPS * 120 s = 3,600 frames = 3,600 * N vectors = 7.2 * 10^7 vectors
# vectors in one video, and each one of these has to be compared with
# all of the vectors from another video (i.e. also 7.2 * 10^7 vectors).
def results_b(num_vectors):
	'''
	Produces query times of exhaustive search over all elements in a dataset for
	datasets with 'num_vectors' 128-dimensional vectors.
	:param num_vectors: array of 'num_vectors' contained in queried datasets

	:returns query times corresponding to the given counts of 'num_vectors'
	'''
	times = []
	for n in num_vectors:
		time = time_exhaustive_search(n, 128)
		times.append(time)
		print(f'Generating results for 1.b): {n:5d}/{num_vectors[-1]:5d}\tTime: {time:5.2f}', end='\r')
	print(70 * '-')
	print('Done. Results for 1.b) generated.')

	return times



if __name__ == '__main__':

	dimensions = [x for x in range(1, 500, 10)]
	# Plot 1.a) to file
	plt.figure(0)
	plt.plot(dimensions, results_a(dimensions))
	plt.title('Query Times')
	plt.xlabel('dimension (D)')
	plt.ylabel('time (ms)')
	plt.savefig('1_1_a.png', bbox_inches='tight')


	num_vectors = [x for x in range(1, 20_000, 500)]
	plt.figure(1)
	plt.plot(num_vectors, results_b(num_vectors))
	plt.title('Query Times')
	plt.xlabel('number of vectors (N)')
	plt.ylabel('time (ms)')
	plt.savefig('1_1_b.png', bbox_inches='tight')





	
