import pandas as pd
import itertools
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import random


def plot_dots_3d(x_points, y_points, z_points, labels):
  fig = plt.figure()
  ax = plt.axes(projection="3d")
  ax.set_xlabel(labels[0])
  ax.set_ylabel(labels[1])
  ax.set_zlabel(labels[2])
  ax.scatter3D(x_points, y_points, z_points);

  plt.show()


def task_1():
  f = pd.read_csv('./datasets/reglab1.txt', delimiter='\t')
  data = np.array(f[list(f)])

  x = data[:, 1]
  y = data[:, 2]
  z = data[:, 0]

  plot_dots_3d(x, y, z, ['x', 'y', 'z'])

  print('<dependent variable> : <accuracy>')
  regrZ = LinearRegression()
  cvZ = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'Z : {cross_val_score(regrZ,list(zip(x,y)), z, cv=cvZ).mean()}')

  regrY = LinearRegression()
  cvY = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'Y : {cross_val_score(regrZ,list(zip(x,z)), y, cv=cvY).mean()}')

  regrX = LinearRegression()
  cvX = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'X : {cross_val_score(regrZ,list(zip(z,y)), x, cv=cvX).mean()}')


def reduce_dimensions(data, feature_name):
  return data.drop(feature_name, axis=1)


def task_2():
  f = pd.read_csv('./datasets/reglab.txt', delimiter='\t')
  y = f.y.values
  x = f.drop('y', axis=1)
  print('<condition> : <accuracy>')
  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'With all x : {cross_val_score(regr, x, y, cv=cv).mean()}')  

  feature_list = ['x1', 'x2', 'x3', 'x4']
  feature_list_2 = list(itertools.combinations(feature_list, r=2))
  feature_list_3 = list(itertools.combinations(feature_list, r=3))

  for i in feature_list:
    x_reduced = reduce_dimensions(x, i)
    regr = LinearRegression()
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    print(f'Without {i} : {cross_val_score(regr, x_reduced, y, cv=cv).mean()}')

  for i in feature_list_2:
    x_reduced = reduce_dimensions(x, i[0])
    x_reduced = reduce_dimensions(x_reduced, i[1])
    regr = LinearRegression()
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    print(f'Without {i[0]}, {i[1]} : {cross_val_score(regr, x_reduced, y, cv=cv).mean()}')

  for i in feature_list_3:
    x_reduced = reduce_dimensions(x, i[0])
    x_reduced = reduce_dimensions(x_reduced, i[1])
    x_reduced = reduce_dimensions(x_reduced, i[2])
    regr = LinearRegression()
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    print(f'Without {i[0]}, {i[1]}, {i[2]} : {cross_val_score(regr, x_reduced, y, cv=cv).mean()}')

  plot_dots_3d(x_reduced.x1.values, x_reduced.x2.values, y, ['x1', 'x2', 'y'])  


def task_3():
  data = pd.read_csv("datasets/cygage.txt", delimiter="\t").values
  x, y = data[:, 1:], data[:, 0]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
  x_train, weights_train = x_train[:, 0], x_train[:, 1]
  x_test, weights_test = x_test[:, 0], x_test[:, 1]

  model = LinearRegression()
  model.fit(x_train.reshape(-1, 1), y_train, sample_weight=weights_train)
  print(f'With weights: accuracy = {model.score(x_test.reshape(-1, 1), y_test, sample_weight=weights_test)}')
  plt.plot(x[:, 0], y, 'o')
  x = x[:, 0]
  plt.plot(x, model.predict(x.reshape(-1, 1)))

  model = LinearRegression()
  model.fit(x_train.reshape(-1, 1), y_train)
  print(f'Without weights: accuracy = {model.score(x_test.reshape(-1, 1), y_test, sample_weight=weights_test)}')
  plt.plot(x, model.predict(x.reshape(-1, 1)))

  plt.show()


def task_4():
  f = pd.read_csv('./datasets/longley.csv')
  f = f.drop('Population', axis=1)
  f = f.sample(frac=1)
  
  y = f.Employed.values
  x = f.drop('Employed', axis=1).values

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True)
  regr = LinearRegression()
  regr.fit(X_train, y_train)
  score = regr.score(X_test, y_test)
  print(f'LinearRegression accuracy : {score}')

  alphas = [10**(-3 + 0.2*i) for i in range(26)]
  scores_test = []
  scores_train = []

  for alpha in alphas: 
    rcv = Ridge(alpha=alpha)
    rcv.fit(X_train, y_train)
    scores_test.append(rcv.score(X_test, y_test))
    scores_train.append(rcv.score(X_train, y_train))

  print(f'Ridge regression accuracy : {np.max(scores_test)}')
  plt.plot(alphas, scores_test, 'r')
  plt.plot(alphas, scores_train, 'b')
  plt.legend(['Test data', 'Train data'])
  plt.show()


def task_5():
  f = pd.read_csv('./datasets/eustock.csv')
  timeline = [i for i in range(1860)]
  plt.plot(timeline, f.DAX.values, 'r')
  plt.plot(timeline, f.SMI.values, 'b')
  plt.plot(timeline, f.CAC.values, 'g')
  plt.plot(timeline, f.FTSE.values, 'y')
  plt.legend(['DAX', 'SMI', 'CAC', 'FTSE'])
  plt.show()

  x = np.array(timeline).reshape(-1, 1)

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'DAX linear regression accuracy : {cross_val_score(regr, x, f.DAX.values, cv=cv).mean()}')
  regr.fit(x, f.DAX.values)
  print(f'DAX linear regression coefficient : {regr.coef_[0]}')

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print()
  print(f'SMI linear regression accuracy : {cross_val_score(regr, x, f.SMI.values, cv=cv).mean()}')
  regr.fit(x, f.SMI.values)
  print(f'SMI linear regression coefficient : {regr.coef_[0]}')

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print()
  print(f'CAC linear regression accuracy : {cross_val_score(regr, x, f.CAC.values, cv=cv).mean()}')
  regr.fit(x, f.CAC.values)
  print(f'CAC linear regression coefficient : {regr.coef_[0]}')

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print()
  print(f'FTSE linear regression accuracy : {cross_val_score(regr, x, f.FTSE.values, cv=cv).mean()}')
  regr.fit(x, f.FTSE.values)
  print(f'FTSE linear regression coefficient : {regr.coef_[0]}')

  # regr = LinearRegression()
  # cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  # print()
  # print(f'All linear regression accuracy : {cross_val_score(regr, x, f.values, cv=cv).mean()}')
  # regr.fit(x, f.values)
  

def preprocces_quarters(arr):
  res = []
  for elem in arr:
    if elem.endswith('Q1'):
      res.append(float(elem[:4]))
    if elem.endswith('Q2'):
      res.append(float(elem[:4] + '.25'))
    if elem.endswith('Q3'):
      res.append(float(elem[:4] + '.5'))
    if elem.endswith('Q4'):
      res.append(float(elem[:4] + '.75'))
  return np.array(res).reshape(-1, 1)    

def task_6():
  f = pd.read_csv('./datasets/JohnsonJohnson.csv')
  my_xticks = f['index']
  plt.xticks(f.index.values, my_xticks, rotation=90)
  plt.plot(f['index'], f.value, 'r')
  plt.show()
  x = f['index'].values
  y = f.value.values

  Q1_years_list = [f"{i} Q1" for i in range(1960, 1981)]
  Q2_years_list = [f"{i} Q2" for i in range(1960, 1981)]
  Q3_years_list = [f"{i} Q3" for i in range(1960, 1981)]
  Q4_years_list = [f"{i} Q4" for i in range(1960, 1981)]

  fQ1 = f.loc[f['index'].isin(Q1_years_list)]
  fQ2 = f.loc[f['index'].isin(Q2_years_list)]
  fQ3 = f.loc[f['index'].isin(Q3_years_list)]
  fQ4 = f.loc[f['index'].isin(Q4_years_list)]

  xQ1 = fQ1['index'].values
  xQ1 = preprocces_quarters(xQ1)
  yQ1 = fQ1.value.values

  xQ2 = fQ2['index'].values
  xQ2 = preprocces_quarters(xQ2)
  yQ2 = fQ2.value.values

  xQ3 = fQ3['index'].values
  xQ3 = preprocces_quarters(xQ3)
  yQ3 = fQ3.value.values

  xQ4 = fQ4['index'].values
  xQ4 = preprocces_quarters(xQ4)
  yQ4 = fQ4.value.values

  x = preprocces_quarters(x)

  print('Accuracy:')
  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'Q1 : {cross_val_score(regr, xQ1, yQ1, cv=cv).mean()}')

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'Q2 : {cross_val_score(regr, xQ2, yQ2, cv=cv).mean()}')

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'Q3 : {cross_val_score(regr, xQ3, yQ3, cv=cv).mean()}')

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'Q4 : {cross_val_score(regr, xQ4, yQ4, cv=cv).mean()}')

  regr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'All quarters : {cross_val_score(regr, x, y, cv=cv).mean()}')

  regr1 = LinearRegression().fit(xQ1, yQ1)
  regr2 = LinearRegression().fit(xQ2, yQ2)
  regr3 = LinearRegression().fit(xQ3, yQ3)
  regr4 = LinearRegression().fit(xQ4, yQ4)
  regrAll = LinearRegression().fit(x, y)

  print('\nCoefficients:')
  print(f'Q1 : {regr1.coef_[0]}')
  print(f'Q2 : {regr2.coef_[0]}')
  print(f'Q3 : {regr3.coef_[0]}')
  print(f'Q4 : {regr4.coef_[0]}')
  print(f'All quarters : {regrAll.coef_[0]}')

  print('\nPrediction for 2016:')
  print(f'Q1 : {regr1.predict([[2016]])[0]}')
  print(f'Q2 : {regr2.predict([[2016.25]])[0]}')
  print(f'Q3 : {regr3.predict([[2016.5]])[0]}')
  print(f'Q4 : {regr4.predict([[2016.75]])[0]}')
  print(f'Year : {regrAll.predict([[2016]])[0]}')
  

def task_7():
  f = pd.read_csv('./datasets/cars.csv')
  speed = f.speed.values
  dist = f.dist.values
  plt.xlabel('speed')
  plt.ylabel('distance')
  plt.scatter(speed, dist, marker='.', c='red')
  plt.show()

  x_train, x_test, y_train, y_test = train_test_split(speed, dist, test_size=0.3, shuffle=True)

  model = Ridge(alpha=0.003)
  model.fit(x_train.reshape(-1, 1), y_train)
  print(f'Accuracy = {model.score(x_test.reshape(-1, 1), y_test)}')
  print(f'Distance for 40 miles/hour = {model.predict([[40]])[0]}')


def task_8():
  f = pd.read_csv('./datasets/svmdata6.txt', delimiter='\t')
  x = np.array(f.X.values).reshape(-1, 1)
  y = f.Y.values

  plt.plot(x, y, '.r')
  plt.show()

  scoring = 'neg_mean_squared_error'
  epsilons = [i for i in np.arange(0, 3.0, 0.01)]
  scores = []
  for epsilon in epsilons:
    svr = SVR(kernel='rbf', C=1.0, epsilon=epsilon)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores.append((cross_val_score(svr, x, y, cv=cv, scoring=scoring).mean()))

  print('Best example:')
  bets_el_index = np.argmax(scores)
  print(f'epsilon = {epsilons[bets_el_index]}; accuracy = {scores[bets_el_index]}')

  plt.plot(epsilons, scores, '.m')
  plt.show()


def task_9():
  f = pd.read_csv('./datasets/nsw74psid1.csv')
  y = f.re78.values
  x = f.drop('re78', axis=1).values

  print('Default parametrs:')
  dtr = DecisionTreeRegressor()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'DecisionTreeRegressor accuracy = {cross_val_score(dtr, x, y, cv=cv).mean()}')

  print()
  lr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'LinearRegression accuracy = {cross_val_score(lr, x, y, cv=cv).mean()}')

  print()
  svr = SVR()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  print(f'SVR accuracy = {cross_val_score(svr, x, y, cv=cv).mean()}')

  print('\nTuned parametrs:')
  dtr = DecisionTreeRegressor(criterion='mae', min_impurity_decrease=25)
  cv = ShuffleSplit(n_splits=5, test_size=0.3)
  print(f'DecisionTreeRegressor accuracy = {cross_val_score(dtr, x, y, cv=cv).mean()}')

  print()
  lr = LinearRegression()
  cv = ShuffleSplit(n_splits=5, test_size=0.3)
  print(f'LinearRegression accuracy = {cross_val_score(lr, x, y, cv=cv).mean()}')

  print()
  svr = SVR(kernel='rbf', C=300, epsilon=0.1)
  cv = ShuffleSplit(n_splits=5, test_size=0.3)
  print(f'SVR accuracy = {cross_val_score(svr, x, y, cv=cv).mean()}')


def main():
  print('Task 1:')
  task_1()
  print('Task 2:')
  task_2()
  print('Task 3:')
  task_3()
  print('Task 4:')
  task_4()
  print('Task 5:')
  task_5()
  print('Task 6:')
  task_6()
  print('Task 7:')
  task_7()
  print('Task 8:')
  task_8()
  print('Task 9:')
  task_9()


if __name__ == "__main__":
  main()