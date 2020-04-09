import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import torch



# device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'
print(f"Operated on: {device}")


def get_data(file_name):
    f = pd.read_csv(file_name)
    state_bin = f[list(f)[:-1]]
    x = np.array(state_bin)

    y = f[list(f)[-1]]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return x, y


def plot_dots(x, y):
    X_y = zip(X, y)
    X_0 = []
    X_1 = []
    for pare in X_y:
        if (pare[1] == 0):
            X_0.append(pare[0])
        else:
            X_1.append(pare[0]) 

    X_0_1 = []
    X_0_2 = []
    X_1_1 = []
    X_1_2 = []

    for i in X_0:
        X_0_1.append(i[0])
        X_0_2.append(i[1])

    for i in X_1:
        X_1_1.append(i[0])
        X_1_2.append(i[1])


    plt.plot(X_0_1, X_0_2, 'o')  
    plt.plot(X_1_1, X_1_2, 'o')  
    plt.show()  


x, y = get_data('datasets/nn_0.csv')

# plot_dots(x, y)

x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

print(type(x_train), type(x_train_tensor), x_train_tensor.type())

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print(a, b)

import torch.nn.functional as F

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

loss_func = F.cross_entropy