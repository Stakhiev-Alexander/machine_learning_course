import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def get_data(file_name):
    f = pd.read_csv(file_name)
    state = f[list(f)[1:-1]]
    state_bin = pd.get_dummies(state)
    x = np.array(state_bin)

    y = f[list(f)[-1]]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    return x, y


def train(x, y, n_neighbors, dist_name):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=dist_name)
    neigh.fit(x, y)

    return neigh


def evaluate_predictions(test_y_data, predictions):
    bac = balanced_accuracy_score(test_y_data, predictions)

    print(f"Unbalanced_accuracy = {accuracy_score(test_y_data, predictions)}")
    print(f"Balanced_accuracy = {balanced_accuracy_score(test_y_data, predictions)}")
    print("\nConfusion matrix:")
    print(confusion_matrix(test_y_data, predictions))
    print(classification_report(test_y_data, predictions))
    return bac


def plot(n_neighbors_arr, accuracy_score_arr, x_axis_name):
    plt.xlabel(x_axis_name)
    plt.ylabel("Accuracy score")

    plt.bar(n_neighbors_arr, accuracy_score_arr)
    plt.show()


def glass_example():
    n_neighbors_arr = []
    n_neighbors_accuracy_score_arr = []

    x, y = get_data("datasets/glass.csv")

    train_x_data = []
    train_y_data = []
    test_x_data = []
    test_y_data = []

    train_x_data, test_x_data, train_y_data, test_y_data = train_test_split(x, y, test_size=0.33)

    # a
    dist_name = 'minkowski'
    for n_neighbors in range(1, 10):
        print(f'n_neighbors = {n_neighbors}')
        n_neighbors_arr.append(n_neighbors)
        neigh = train(train_x_data, train_y_data, n_neighbors, dist_name)
        predictions = neigh.predict(test_x_data)
        n_neighbors_accuracy_score_arr.append(evaluate_predictions(test_y_data, predictions))

    # b
    n_neighbors = 3
    dist_names = ('euclidean', 'manhattan', 'chebyshev', 'minkowski')
    dist_names_accuracy_score_arr = []

    for dist_name in dist_names:
        neigh = train(train_x_data, train_y_data, n_neighbors, dist_name)
        predictions = neigh.predict(test_x_data)
        dist_names_accuracy_score_arr.append(evaluate_predictions(test_y_data, predictions))

    # c
    n_neighbors = 3
    dist_name = 'minkowski'
    neigh = train(train_x_data, train_y_data, n_neighbors, dist_name)
    predictions = neigh.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])

    print(
        'Type of glass with features: RI =1.516 Na =11.7 Mg =1.01 Al =1.19 Si =72.59 K=0.43 Ca =11.44 Ba =0.02 Fe =0.1')
    print(predictions[0])

    plot(dist_names, dist_names_accuracy_score_arr, 'DistanceMetric')
    plot(n_neighbors_arr, n_neighbors_accuracy_score_arr, 'Number of neighbors')
