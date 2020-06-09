from __future__ import print_function
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-5
def log(x):
    return np.log(x + eps)

class LogisticRegression():
    def __init__(self):
        self.weights = np.array([])
        self.losses = []  
        self.lr = 1e-1
        self.max_iter = 10 
        
    def init_weights(self, dim):        
        # uniform initialization of weights
        self.weights = np.ones((dim,1)) / dim
    
    def predict_proba(self, features):
        """
        Exercise 1a: Compute the probability of assigning a class to each feature of an image
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
        Returns:
            prob (np.array): probabilities [N] of N examples
        """
    		# TODO: INSERT
        linear = np.dot(features, self.weights)
        prob = (1 / (1 + np.exp(-linear)))
        return prob
    
    def predict(self, features):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
        Returns:
            pred (np.array): predictions [N] of N examples
        """
        prob = self.predict_proba(features)
        # decision boundary at 0.5
        pred = np.array([ 1.0 if x >= 0.5 else 0.0 for x in prob])[:,np.newaxis]
        return pred
    
    def compute_loss(self, features, labels):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
        Returns:
            loss (scalar): loss of the current model
        """
        examples = len(labels)
        
        '''
        Exercise 1b:    Compute the loss for the features of all input images
                        NOTE: Don't forget to remove the first quit() command in the main program!

        HINT: Use the provided log function to avoid nans with large learning rate
        '''
        # loss = 0 # TODO: REPLACE

        prob_pullover = self.predict_proba(features)
        prob_coat = 1 - prob_pullover
        loss = -(labels * log(prob_pullover)) - ((1 - labels) * log(prob_coat))
        
        return loss.sum() / examples
            
    def score(self, pred, labels):
        """
        Args:
            pred (np.array): predictions [N, 1] of N examples
            labels (np.array): labels [N, 1] of N examples
        Returns:
            score (scalar): accuracy of the predicted labels
        """
        diff = pred - labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))
    
    def update_weights(self, features, labels, lr):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
            lr (scalar): learning rate scales the gradients
        """
        examples = len(labels)
        
        '''
        Exercise 1c:    Compute the gradients given the features of all input images
                        NOTE: Don't forget to remove the second quit() command in the main program!
        '''
        # gradient = 0 # TODO: REPLACE

        gradient = np.sum((self.predict_proba(features) - labels) * features, axis=0)[:, np.newaxis]
        
        # update weights
        self.weights -= lr * gradient / examples 
        
    def fit(self, features, labels):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
        """
        # gradient descent    
        for i in range(self.max_iter):
            # update weights using the gradients
            self.update_weights(features, labels, self.lr)
            
            # compute loss
            loss = self.compute_loss(features, labels)
            self.losses.append(loss)
            
            # print current loss
            print('Iteration {}\t Loss {}'.format(i, loss))
    

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

# load fashion mnist
train_img, train_label = load_mnist('Exercise_2/FashionMNIST', kind='train', each=1)
test_img, test_label = load_mnist('Exercise_2/FashionMNIST', kind='t10k', each=1)
train_img = train_img.astype(np.float)/255.
test_img = test_img.astype(np.float)/255.

# label definition of fashion mnist
labels = { 0: 'T-shirt/top',
           1: 'Trouser',
           2: 'Pullover',
           3: 'Dress',
           4: 'Coat',
           5: 'Sandal',
           6: 'Shirt',
           7: 'Sneaker',
           8: 'Bag',
           9: 'Ankle boot'}

# consider only the classes 'Pullover' and 'Coat'
labels_mask = [2, 4]
train_mask = np.zeros(len(train_label), dtype=bool)
test_mask = np.zeros(len(test_label), dtype=bool)
train_mask[(train_label == labels_mask[0]) | (train_label == labels_mask[1])] = 1
test_mask[(test_label == labels_mask[0]) | (test_label == labels_mask[1])] = 1

# classification of Pullover
train_img = train_img[train_mask,:]
test_img = test_img[test_mask,:]
train_label = np.array([ 1.0 if x == labels_mask[0] else 0.0 for x in train_label[train_mask]])[:,np.newaxis]
test_label = np.array([ 1.0 if x == labels_mask[0] else 0.0 for x in test_label[test_mask]])[:,np.newaxis]

# init logistic regression
logreg = LogisticRegression()
logreg.init_weights(train_img.shape[1])
logreg.lr = 1e-2
logreg.max_iter = 10

accs = []

# testing without training
y_pred = logreg.predict(test_img)
score = logreg.score(y_pred, test_label)
accs.append(score)
print('Accuracy of initial logistic regression classifier on test set: {:.2f}'.format(score))

# quit() ### Exercise 1b: Remove exit ### 
            
# compute initialization loss
loss = logreg.compute_loss(train_img, train_label)
print('Initialization loss {}'.format(loss))

# quit() ### Exercise 1c: Remove exit ###

'''
Exercise 1d: Plot the cross entropy loss for t=1 and t=1
'''
#TODO: Insert

f_w = np.linspace(0, 1, 200)
loss_t_0 = -log(1 - f_w)
loss_t_1 = -log(f_w)
plt.figure(0)
plt.plot(f_w, loss_t_0, c='b', label='loss for t = 0')
plt.plot(f_w, loss_t_1, c='r', label='loss for t = 1')
plt.title('Cross entropy loss')
plt.xlabel('f_w')
plt.ylabel('loss')
plt.legend()
plt.savefig('2_1_d.png', bbox_inches='tight')



# compute test error after max_iter
for i in range(0,100):
    # training
    logreg.fit(train_img, train_label)
    
    # testing
    y_pred = logreg.predict(test_img)
    score = logreg.score(y_pred, test_label)
    accs.append(score)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(score))
 
'''
Exercise 1e: Plot the learning curves (losses and accs) using different learning rates (1e-4,1e-3,1e-2,1e-1,1e-0)
'''
losses = logreg.losses
TODO: INSERT
fig, ax1 = plt.subplots()
fig.suptitle(f'Learning curve of lr = {logreg.lr}')

ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax1.plot(np.arange(len(losses)), losses, c='b', label='loss')

# Add the second axis with another scale for accuracy
ax2 = ax1.twinx()

ax2.set_ylabel('accuracy')
# Account for the initial accuracy
ax2.plot(np.arange(0, 1_000, 10), accs[1:], c='r', label='accuracy')

fig.legend()
fig.savefig(f'2_1_e_{logreg.lr}.png', bbox_inches='tight')


'''
Exercise 1f: Plot the optimized weights and weights.*img (.* denotes element-wise multiplication)
'''
# TODO: 


# plt.figure(2)
# plt.imshow(logreg.weights.reshape(28, 28))
# plt.savefig('2_1_f.png', bbox_inches='tight')


# img_class_1 = test_img[np.where(test_label == 0)[0][0:5]].reshape(5, 28, 28)
# img_class_2 = test_img[np.where(test_label == 1)[0][0:5]].reshape(5, 28, 28)

# fig, ax = plt.subplots(5, 2)
# fig.figsize = (15, 5)
# print(img_class_1[0].reshape(1, -1).shape)
# for i in range(5):
#     ax[i,0].imshow(logreg.weights.reshape(28, 28) * img_class_1[i])
#     ax[i,0].set_title(logreg.predict_proba(img_class_1[i].reshape(1, -1)))
#     ax[i,1].imshow(logreg.weights.reshape(28, 28) * img_class_2[i])
#     ax[i,1].set_title(logreg.predict_proba(img_class_2[i].reshape(1, -1)))

# plt.savefig('2_1_f_1.png', bbox_inches='tight')
