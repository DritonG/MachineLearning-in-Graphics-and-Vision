'''
group member 1: Driton Goxhufi      xxxxxxx     driton.goxhufi@student.uni-tuebingen.de
group member 2: Damir Ravlija       5503184     damir.ravlija@student.uni-tuebingen.de
'''

import numpy as np
from sklearn.neighbors import KDTree

import problem_1_2_a

# 'fashion_mnist' should be in the working directory
train_img, train_label = problem_1_2_a.load_mnist('fashion_mnist', kind='train', each=10)
test_img, test_label = problem_1_2_a.load_mnist('fashion_mnist', kind='t10k', each=10)

train_ind_c2_c6 = np.where(np.logical_or(train_label==2, train_label==6))
train_img_c2_c6, train_label_c2_c6 = train_img[train_ind_c2_c6], train_label[train_ind_c2_c6]

test_ind_c2_c6 = np.where(np.logical_or(test_label==2, test_label==6))
test_img_c2_c6, test_label_c2_c6 = test_img[test_ind_c2_c6], test_label[test_ind_c2_c6]

tree = KDTree(train_img_c2_c6)
nearest_nbrs = tree.query(test_img_c2_c6, return_distance=False)

tp_tn_ind = np.where(test_label_c2_c6 == train_label_c2_c6[nearest_nbrs].flatten())
fp_fn_ind = np.where(test_label_c2_c6 != train_label_c2_c6[nearest_nbrs].flatten())

true_pos_2 = np.count_nonzero(test_label_c2_c6[tp_tn_ind]==2)
true_pos_6 = len(test_label_c2_c6[tp_tn_ind]) - true_pos_2

# All that were predicted 2, but were not correct
false_pos_2 = np.count_nonzero(train_label_c2_c6[nearest_nbrs][fp_fn_ind]==2)
false_pos_6 = len(fp_fn_ind[0])-false_pos_2

precision_2 = true_pos_2 / (true_pos_2 + false_pos_2)
precision_6 = true_pos_6 / (true_pos_6 + false_pos_6)

recall_2 = true_pos_2 / (true_pos_2 + false_pos_6)
recall_6 = true_pos_6 / (true_pos_6 + false_pos_2)

print('Precision (with "Pullover" (2) as positive): ', precision_2)
print('Precision (with "Shirt" (6) as positive): ', precision_6)
print(60*'-')
print('Recall (with "Pullover" (2) as positive): ', recall_2)
print('Recall (with "Shirt" (6) as positive): ', recall_6)
