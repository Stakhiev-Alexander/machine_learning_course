import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def plot(x, y):
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy score")

    plt.plot(x, y)
    plt.show()

def task_1():
  data = read_csv("datasets/glass.csv", delimiter=",").values
  X, y = data[:, :-1], data[:, -1]

  n_estimators_arr = [i for i in range(1,100)]
  scores = []

  for n_estimators in n_estimators_arr:
    print(n_estimators)
    model = BaggingClassifier(base_estimator=LinearSVC(), n_estimators=n_estimators).fit(X, y)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores.append(cross_val_score(model, X, y, cv=cv).mean())

  plot(n_estimators_arr, scores)


def task_2():
  data = read_csv("datasets/vehicle.csv", delimiter=",").values
  X, y = data[:, :-1], data[:, -1]

  n_estimators_arr = [i for i in range(1,100)]
  # n_estimators_arr = [3, 50]
  scores = []

  for n_estimators in n_estimators_arr:
    print(f"Number of estimators = {n_estimators}")
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators).fit(X, y)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores.append(cross_val_score(model, X, y, cv=cv).mean())
    # print(f"Score = {cross_val_score(model, X, y, cv=cv).mean()}")
  
  plot(n_estimators_arr, scores)


def encode_labels(data):
  # data = pd.get_dummies(data, columns=['Sex'])
  # data = pd.get_dummies(data, columns=['Embarked'])

  dicts = {}
  label = LabelEncoder()

  label.fit(data.Sex.drop_duplicates())
  dicts['Sex'] = list(label.classes_)
  data.Sex = label.transform(data.Sex)

  label.fit(data.Embarked.drop_duplicates())
  dicts['Embarked'] = list(label.classes_)
  data.Embarked = label.transform(data.Embarked)

  return data


def get_titanic_dataset_data(file_name):
  data = read_csv(file_name, delimiter=',').drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
  data.Age = data.Age.fillna(data.Age.mean())
  data.Embarked =  data.Embarked.fillna('S')
  
  data = encode_labels(data)

  y = data.Survived.values
  x = data.drop('Survived', axis=1).values
  return x, y


def task_3():
  X_train, y_train = get_titanic_dataset_data('datasets/titanic.csv')
  estimators = [
    ('rf', RandomForestClassifier(n_estimators=500, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
  ]
  model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()).fit(X_train, y_train)
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(cross_val_score(model, X_train, y_train, cv=cv).mean())


# task_1()
# task_2()
task_3()