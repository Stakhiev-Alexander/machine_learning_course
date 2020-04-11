import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def get_data(file_name):
    f = pd.read_csv(file_name)
    state = f[list(f)[:-1]]
    state_bin = pd.get_dummies(state)
    x = np.array(state_bin)

    y = f[list(f)[-1]]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    return x, y


def train(x, y, test_data_coef):
    train_x_data = []
    train_y_data = []
    test_x_data = []
    test_y_data = []

    train_x_data, test_x_data, train_y_data, test_y_data = train_test_split(x, y, test_size=test_data_coef)

    model = GaussianNB()
    model.fit(train_x_data, train_y_data)
    predictions = model.predict(test_x_data)

    return test_y_data, predictions


def evaluate_predictions(test_y_data, predictions):
    bac = balanced_accuracy_score(test_y_data, predictions)

    print(f"Unbalanced_accuracy = {accuracy_score(test_y_data, predictions)}")
    print(f"Balanced_accuracy = {balanced_accuracy_score(test_y_data, predictions)}")
    print("\nConfusion matrix:")
    print(confusion_matrix(test_y_data, predictions))
    print(classification_report(test_y_data, predictions))
    return bac


def plot(test_data_coef_arr, accuracy_score_arr):
    plt.xlabel("Test data coefficient")
    plt.ylabel("Accuracy score")

    plt.plot(test_data_coef_arr, accuracy_score_arr)
    plt.show()


def tic_tac_toe_example():
    test_data_coef_arr = []
    accuracy_score_arr = []

    x, y = get_data("datasets/tic_tac_toe.txt")

    for test_data_coef in np.arange(0.05, 0.95, 0.05):
        test_data_coef_arr.append(test_data_coef)
        test_y_data, predictions = train(x, y, test_data_coef)
        accuracy_score_arr.append(evaluate_predictions(test_y_data, predictions))

    plot(test_data_coef_arr, accuracy_score_arr)


def spam_example():
    test_data_coef_arr = []
    accuracy_score_arr = []

    x, y = get_data("datasets/spam.csv")

    for test_data_coef in np.arange(0.05, 0.95, 0.01):
        test_data_coef_arr.append(test_data_coef)
        test_y_data, predictions = train(x, y, test_data_coef)
        accuracy_score_arr.append(evaluate_predictions(test_y_data, predictions))

    plot(test_data_coef_arr, accuracy_score_arr)


def roc_curve_plot(test_y_data, y_score):
    plt.figure(figsize=(6, 5))
    fpr, tpr, thresholds = roc_curve(test_y_data, y_score[:, 1], pos_label=1)
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')


def dots_example():
    test_data_coef_arr = []
    accuracy_score_arr = []

    x = []
    y = []
    x0_arr_class_1 = []
    x1_arr_class_1 = []
    x0_arr_class_2 = []
    x1_arr_class_2 = []

    for i in range(80):
        x0 = np.random.normal(13, 2)
        x1 = np.random.normal(20, 2)
        x0_arr_class_1.append(x0)
        x1_arr_class_1.append(x1)
        x.append([x0, x1])
        y.append(-1)

    for i in range(20):
        x0 = np.random.normal(20, 2)
        x1 = np.random.normal(4, 2)
        x0_arr_class_2.append(x0)
        x1_arr_class_2.append(x1)
        x.append([x0, x1])
        y.append(1)

    plt.xlabel("x0")
    plt.ylabel("x1")

    plt.plot(x0_arr_class_1, x1_arr_class_1, 'o')
    plt.plot(x0_arr_class_2, x1_arr_class_2, 'o')
    # plt.show()

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    test_data_coef = 0.33

    train_x_data = []
    train_y_data = []
    test_x_data = []
    test_y_data = []

    train_x_data, test_x_data, train_y_data, test_y_data = train_test_split(x, y, test_size=test_data_coef)

    model = GaussianNB()
    y_score = model.fit(train_x_data, train_y_data).predict_proba(test_x_data)
    predictions = model.predict(test_x_data)

    evaluate_predictions(test_y_data, predictions)

    roc_curve_plot(test_y_data, y_score)
    disp = plot_precision_recall_curve(model, test_x_data, test_y_data)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(0.01))

    plt.show()
