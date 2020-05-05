'''
group member 1: Driton Goxhufi      xxxxxxx     driton.goxhufi@student.uni-tuebingen.de
group member 2: Damir Ravlija       5503184     damir.ravlija@student.uni-tuebingen.de
'''

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

import timeit

import problem_1_1_a_b

def kdtree_search(tree, queries):
    start = timeit.default_timer()

    tree.query(dataset)
        
    end = timeit.default_timer()
    
    time = end - start

    return time  


if __name__ == '__main__':
    # timings for different D
    times = []

    # kd-tree search
    dimensions = list(range(1, 500, 10))
    for D in dimensions:
        dataset = problem_1_1_a_b.create_dataset(1024, D)
        # TODO
        tree = KDTree(dataset)
        time = kdtree_search(tree, dataset)
        times.append(time)
        print(f'Generating results for 1.c): {D:3d}/{dimensions[-1]:3d}\tTime: {time:4.2f}', end='\r')

    print(60 * '-')
    print('Done. Results for 1.c) generated.')


    # plot to file
    plt.plot(range(1, 500, 10), times)
    plt.title('Query Times')
    plt.xlabel('dimension (D)')
    plt.ylabel('time (ms)')
    plt.savefig('1_1_c.png', bbox_inches='tight')