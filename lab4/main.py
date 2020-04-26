import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

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

# task_1()
task_2()