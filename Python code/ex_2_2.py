import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
meas = load_iris().data
X = meas[:, [2, 3]]

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5).fit(X)
IDX = dbscan.labels_

import matplotlib.pyplot as plt

plt.figure(1)
plt.scatter(X[:,0],X[:,1], c=IDX)
plt.show()

print("Silhouette score(Pre normalization): ",metrics.silhouette_score(X, IDX))


from scipy.stats import zscore

xV1 = zscore(X[:,0])
xV2 = zscore(X[:,1])
X = np.array([xV1, xV2]).T  # [xV1, xV2] Έγινε αντικατάσταση της συγκεκριμένης γραμμής για να
                            # πάρουμε το σωστο πληθος idx

dbscan = DBSCAN(eps=0.1, min_samples=5).fit(X)
IDX = dbscan.labels_
plt.figure(1)
plt.scatter(xV1, xV2, c=IDX)
plt.show()

print("Silhouette score(Post normalization): ",metrics.silhouette_score(X, IDX))