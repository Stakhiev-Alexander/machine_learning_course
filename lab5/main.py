import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def get_data_1(file_name):
    f = pd.read_csv(file_name, delimiter='\t')
    x = np.array(f[list(f)])
    return x


def plot_dots_3d(x_points, y_points, z_points):
  fig = plt.figure()
  ax = plt.axes(projection="3d")
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.scatter3D(x_points, y_points, z_points);

  plt.show()


data = get_data_1('./datasets/reglab1.txt') 
# plot_dots_3d(x[:,1], x[:,2], x[:,0])


# Split the data into training/testing sets
X_train = data[:-40, 1:2]
X_test = data[-40:, 1:2]

# Split the targets into training/testing sets
y_train = data[:-40, 0]
y_test = data[-40:, 0]

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter3D(X_test[:, 0], X_test[:, 1], y_test);

# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

plt.show()