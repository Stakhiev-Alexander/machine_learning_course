from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib.image as img
from scipy import misc 
from matplotlib.pyplot import imread

def plot_results(X, y, centers):
  # Plot the clustered data
  plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

  if (len(centers) > 0):
      plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
  plt.show()

def get_data_2(file_name):
    f = pd.read_csv(file_name, sep='\t', engine='python')
    state = f[list(f)]
    x = np.array(state)
    return x

def get_data_4(file_name):
    f = pd.read_csv(file_name)
    state = f[list(f)]
    state_bin = pd.get_dummies(state)
    x = np.array(state_bin)
    return x

def task_2_subtask(file_name):
  X=get_data_2(file_name)

  # Kmeans
  for k in range(1, 5):
      km = KMeans(n_clusters=k, random_state=0).fit(X)
      y = km.predict(X)
      centers = km.cluster_centers_
      plot_results(X, y, centers)

  #DBSCAN
  y = DBSCAN().fit_predict(X)
  centers = []
  print(y)
  plot_results(X, y, centers)

  # AgglomerativeClustering
  for k in range(1, 5):
      y = AgglomerativeClustering(n_clusters=k).fit_predict(X)
      centers = []
      print(y)
      plot_results(X, y, centers)

def task_2():
  task_2_subtask('./datasets/clustering_1.csv')
  task_2_subtask('./datasets/clustering_2.csv')
  task_2_subtask('./datasets/clustering_3.csv')

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def task_4():
  x = get_data_4('./datasets/votes.csv')


  # find procent of NaN in dataset
  nan_count, count = 0, 0

  for el in x:
      for el2 in el:
          if (np.isnan(el2)):
              nan_count += 1
          count +=1   

  print(f'Procent of NaNs in dataset: {nan_count/count}')

  # replace NaN with 0
  # x = np.nan_to_num(x)

  # replace NaN with mean
  col_mean = np.nanmean(x, axis=0)
  inds = np.where(np.isnan(x))
  x[inds] = np.take(col_mean, inds[1])

  model = AgglomerativeClustering(n_clusters=2)

  model = model.fit(x)
  plt.title('Hierarchical Clustering Dendrogram for votes.csv')
  plot_dendrogram(model, labels=model.labels_)
  plt.show()


def task_1():
	return "olo"


task_1()
task_2()
task_4()
