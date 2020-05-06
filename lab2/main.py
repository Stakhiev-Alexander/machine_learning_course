import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


N = 50 # number of points per class
D = 2 # dimensionality
K = 2 # number of classes


device = torch.device('cuda')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)


def get_data(file_name):
    f = pd.read_csv(file_name)
    state_bin = f[list(f)[:-1]]
    x = np.array(state_bin)

    y = f[list(f)[-1]]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return x, y


def get_MNIST_DataLoader():      
    batch_size = 32

    train_dataset = datasets.MNIST('./data', 
                                   train=True, 
                                   download=True, 
                                   transform=transforms.ToTensor())

    validation_dataset = datasets.MNIST('./data', 
                                        train=False, 
                                        transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False)

    return train_loader, validation_loader


def plot_dots(x, y):
    plt.figure(figsize=(8,6))
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.rainbow)
    plt.xlabel("x0", fontsize=15)
    plt.ylabel("x1", fontsize=15)
    plt.show()


def evalute_and_plot_results_1(X, y, W, b):
    scores = np.dot(X, W) + b
    predicted_class = np.argmax(scores, axis=1)
    print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


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


def evalute_and_plot_results_2(X, y, W, b, W2, b2):
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def train_1(X, y):
    # initialize parameters randomly
    W = 0.01 * np.random.randn(D,K)
    b = np.zeros((1,K))

    # some hyperparameters
    learning_rate = 1e-0
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
        W += -learning_rate * dW
        b += -learning_rate * db

    return W, b    


def train_2(X, y):
    # initialize parameters randomly
    h = 100 # size of hidden layer
    W = 0.01 * np.random.randn(D,h)
    b = np.zeros((1,h))
    W2 = 0.01 * np.random.randn(h,K)
    b2 = np.zeros((1,K))

    step_size = 1e-0
    reg = 1e-3 # regularization strength

    # gradient descent loop
    num_examples = X.shape[0]
    for i in range(1000):
        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W) + b) # ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples),y])
        data_loss = np.sum(corect_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        if i % 100 == 0:
            print( "iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

    return W, b, W2, b2


def train_3(epoch, model, train_loader, optimizer, criterion, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def validate(loss_vector, accuracy_vector, model, train_loader, validation_loader, criterion):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))    
   

def main():
    # a
    X, y = get_data('datasets/nn_0.csv')
    W, b = train_1(X, y)
    evalute_and_plot_results_1(X, y, W, b)

    X, y = get_data('datasets/nn_1.csv')
    W, b = train_1(X, y)
    evalute_and_plot_results_1(X, y, W, b)

    #b
    X, y = get_data('datasets/nn_1.csv')
    W, b, W2, b2 = train_2(X, y)
    evalute_and_plot_results_2(X, y, W, b, W2, b2)

    #c
    train_loader, validation_loader = get_MNIST_DataLoader() 
    model = Net().to(device)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    epochs = 10

    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train_3(epoch, model, train_loader, optimizer, criterion)
        validate(lossv, accv, model, train_loader, validation_loader, criterion)


    plt.plot(np.arange(1,epochs+1), lossv)
    plt.plot(np.arange(1,epochs+1), accv)

    plt.show()

main()    
