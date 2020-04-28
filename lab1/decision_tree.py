import graphviz
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def get_data_spam7(file_name):
    f = pd.read_csv(file_name)
    state = f[list(f)[:-1]]
    state_bin = pd.get_dummies(state)
    x = np.array(state_bin)

    y = f[list(f)[-1]]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return x, y


def get_data_glass(file_name):
    f = pd.read_csv(file_name)
    state = f[list(f)[1:-1]]
    state_bin = pd.get_dummies(state)
    x = np.array(state_bin)

    y = f[list(f)[-1]]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return x, y


def evaluate_predictions(test_y_data, predictions):
    bac = balanced_accuracy_score(test_y_data, predictions)

    print(f"Unbalanced_accuracy = {accuracy_score(test_y_data, predictions)}")
    print(f"Balanced_accuracy = {balanced_accuracy_score(test_y_data, predictions)}")
    print("\nConfusion matrix:")
    print(confusion_matrix(test_y_data, predictions))
    print(classification_report(test_y_data, predictions))
    return bac


def train(x, y, feature_names, target_names, res_graph_name):
    train_x_data = []
    train_y_data = []
    test_x_data = []
    test_y_data = []

    train_x_data, test_x_data, train_y_data, test_y_data = train_test_split(x, y, test_size=0.33)

    clf = tree.DecisionTreeClassifier(min_samples_leaf=4)
    clf = clf.fit(train_x_data, train_y_data)
    print(clf.feature_importances_)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render(res_graph_name)

    predictions = clf.predict(test_x_data)
    evaluate_predictions(test_y_data, predictions)


def glass_example_tree():
    x, y = get_data_glass("datasets/glass.csv")
    f = pd.read_csv("datasets/glass.csv")
    feature_names = list(f.columns[1:-1])
    target_names = [str(elem) for elem in np.unique(f[list(f)[-1]])]
    train(x, y, feature_names, target_names, 'glass_graph')


def spam7_example_tree():
    x, y = get_data_spam7("datasets/spam7.csv")
    f = pd.read_csv("datasets/spam7.csv")
    feature_names = list(f.columns[0:-1])
    print(feature_names)
    target_names = [str(elem) for elem in np.unique(f[list(f)[-1]])]
    train(x, y, feature_names, target_names, 'spam7_graph')
