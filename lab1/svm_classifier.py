import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def plot(titles, n, m, models, X, y):
    # Set-up n*m grid for plotting.
    fig, sub = plt.subplots(n, m, figsize=(5, 15))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    X0, X1 = [line[0] for line in X], [line[1] for line in X]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = min(x) - 1, max(x) + 1
    y_min, y_max = min(y) - 1, max(y) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def get_data(file_name):
    f = open(file_name)
    f.readline()
    tmp = [line.replace('\n', '').split('\t')[1:] for line in f]
    X = [[float(line[0]), float(line[1])] for line in tmp]
    y = [line[-1] for line in tmp]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    f.close()

    return X, y


def evaluate_predictions(test_y_data, predictions):
    bac = balanced_accuracy_score(test_y_data, predictions)
    print(f"Unbalanced_accuracy = {accuracy_score(test_y_data, predictions)}")
    print(f"Balanced_accuracy = {balanced_accuracy_score(test_y_data, predictions)}")
    print("\nConfusion matrix:")
    print(confusion_matrix(test_y_data, predictions))
    print(classification_report(test_y_data, predictions))
    return bac


def a_example():
    X, y = get_data('datasets/svmdata_a.txt')
    test_X, test_y = get_data('datasets/svmdata_a_test.txt')

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.00  # SVM regularization parameter
    model_linear = svm.SVC(kernel='linear', C=C)
    model_LinearSVC = svm.LinearSVC(C=C, max_iter=10000)
    model_linear.fit(X, y)
    model_LinearSVC.fit(X, y)

    predictions_linear = model_linear.predict(test_X)
    predictions_LinearSVC = model_LinearSVC.predict(test_X)
    print(f"Support vector number: {model_linear.n_support_[0]}")

    evaluate_predictions(y, model_linear.predict(X))
    evaluate_predictions(test_y, predictions_linear)

    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)')
    plot(titles, 2, 1, (model_linear, model_LinearSVC), X, y)


def plot_single_graph(model_linear, X, y):
    X0, X1 = [line[0] for line in X], [line[1] for line in X]
    xx, yy = make_meshgrid(X0, X1)

    fig, ax = plt.subplots()

    plot_contours(ax, model_linear, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('linear')

    plt.show()


def b_example():
    X, y = get_data('datasets/svmdata_b.txt')
    test_X, test_y = get_data('datasets/svmdata_b_test.txt')
    C_arr = []
    C_accuracy_score_arr_test = []
    C_accuracy_score_arr_train = []

    for C in range(1, 1500):
        C_arr.append(C)

        model_linear = svm.SVC(kernel='linear', C=C)
        model_linear.fit(X, y)

        predictions_linear_test = model_linear.predict(test_X)
        predictions_linear_train = model_linear.predict(X)

        C_accuracy_score_arr_test.append(evaluate_predictions(test_y, predictions_linear_test))
        C_accuracy_score_arr_train.append(evaluate_predictions(y, predictions_linear_train))

    plt.plot(C_arr, C_accuracy_score_arr_test)
    plt.plot(C_arr, C_accuracy_score_arr_train)
    plt.show()
    # plot_single_graph(model_linear, test_X, test_y)
    # plot_single_graph(model_linear, X, y)


def c_example():
    X, y = get_data('datasets/svmdata_c.txt')

    C = 1.00
    models = (svm.SVC(kernel='linear', C=C),
              svm.SVC(kernel='sigmoid', C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=1, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=4, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=5, gamma='auto', C=C))

    models = (clf.fit(X, y) for clf in models)

    titles = ('linear',
              'sigmoid',
              'rbf',
              'poly(degree = 1)',
              'poly(degree = 2)',
              'poly(degree = 3)',
              'poly(degree = 4)',
              'poly(degree = 5)')

    plot(titles, 4, 2, models, X, y)


def d_example():
    X, y = get_data('datasets/svmdata_d.txt')
    X_test, y_test = get_data('datasets/svmdata_d_test.txt')

    C = 1
    models = (svm.SVC(kernel='sigmoid', C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=1, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=4, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=5, gamma='auto', C=C))

    models = (clf.fit(X, y) for clf in models)

    titles = ('sigmoid',
              'rbf',
              'poly(degree = 1)',
              'poly(degree = 2)',
              'poly(degree = 3)',
              'poly(degree = 4)',
              'poly(degree = 5)')

    predictions_test = (model.predict(X_test) for model in models)

    for pred in predictions_test:
        print("start")
        evaluate_predictions(pred, y_test)
        # plot(titles, 4, 2, models, X_test, y_test)


def e_example():
    X, y = get_data('datasets/svmdata_e.txt')
    X_test, y_test = get_data('datasets/svmdata_e_test.txt')

    C = 10000
    models = (svm.SVC(kernel='sigmoid', C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=1, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=4, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=5, gamma='auto', C=C))

    models = (clf.fit(X, y) for clf in models)

    titles = ('sigmoid',
              'rbf',
              'poly(degree = 1)',
              'poly(degree = 2)',
              'poly(degree = 3)',
              'poly(degree = 4)',
              'poly(degree = 5)')

    plot(titles, 4, 2, models, X, y)
    # predictions_test = (model.predict(X_test) for model in models)
    # predictions_train = (model.predict(X) for model in models)

    # for pred in predictions_train:
    #   print("start")
    #   evaluate_predictions(pred, y)
