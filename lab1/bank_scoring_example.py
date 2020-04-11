import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def get_data(file_name):
    f = pd.read_csv(file_name, sep='\t')
    x = f[list(f)[1:]]

    y = f[list(f)[0]]
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


def bank_scoring_example():
    X, y = get_data("datasets/bank_scoring_train.csv")
    X_test, y_test = get_data('datasets/bank_scoring_test.csv')

    print("GaussianNB")
    model = GaussianNB()
    model.fit(X, y)
    predictions = model.predict(X_test)

    evaluate_predictions(y_test, predictions)

    print("kNN")
    n_neighbors = 1
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X, y)
    predictions = neigh.predict(X_test)
    evaluate_predictions(y_test, predictions)

    print("Decision tree")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    predictions = clf.predict(X_test)
    evaluate_predictions(y_test, predictions)
