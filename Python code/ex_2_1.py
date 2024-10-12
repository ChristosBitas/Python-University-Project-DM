
from sklearn import metrics

import scipy.io
import numpy as np
mat_file = scipy.io.loadmat( 'mydata.mat')
X = np.array(mat_file['X'])

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=15).fit(X)
IDX = dbscan.labels_

import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(X[:,0],X[:,1])
plt.show()

plt.figure(1)
plt.scatter(X[:,0],X[:,1], c=IDX)
plt.show()

n_clusters_ = len(set(IDX)) - (1 if -1 in IDX else 0)
n_noise_ = list(IDX).count(-1)

print("Silhouette score: ",metrics.silhouette_score(X, IDX))