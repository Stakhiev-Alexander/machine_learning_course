import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import torch
from torch import nn
import torch.nn.functional as F


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'
print(f"Operated on: {device}")

N = 50 # number of points per class
D = 2 # dimensionality
K = 2 # number of classes

def get_data(file_name):
    f = pd.read_csv(file_name)
    state_bin = f[list(f)[:-1]]
    x = np.array(state_bin)

    y = f[list(f)[-1]]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return x, y


def plot_dots(x, y):
    plt.figure(figsize=(8,6))
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.rainbow)
    plt.xlabel("x0", fontsize=15)
    plt.ylabel("x1", fontsize=15)
    plt.show()



X, y = get_data('datasets/nn_1.csv')

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(100):

    # evaluate class scores, [N x K]
    scores = np.dot(X, W) + b

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print ("iteration %d: loss %f" % (i, loss))

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg*W # regularization gradient

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.xlabel("x0", fontsize=15)
plt.ylabel("x1", fontsize=15)
plt.show()








