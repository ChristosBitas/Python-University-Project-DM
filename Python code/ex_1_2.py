import math
import scipy.io
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

mat_file = scipy.io.loadmat('xV.mat')
xV = np.array(mat_file['xV'])
def classification(idx_0, idx_1, p_flag=False): # Συνάρτηση για την αποφυγή επανάληψης του κώδικα σε κάθε περίπτωση
    X = xV[:,[idx_0, idx_1]]
    k = 3
    kmeans = KMeans(n_clusters=k, n_init=10).fit(X)
    IDX = kmeans.labels_
    C = kmeans.cluster_centers_
    if p_flag:
        plt.plot(X[IDX==0][:,0], X[IDX==0][:,1], 'limegreen', marker='o', linewidth=0, label='C1')
        plt.plot(X[IDX==1][:,0], X[IDX==1][:,1], 'yellow', marker='o', linewidth=0, label='C2')
        plt.plot(X[IDX==2][:,0], X[IDX==2][:,1], 'c.', marker='o', label='C3')
        plt.scatter(C[:,0], C[:,1], marker='x', color='black', s=150 , linewidth=3, label="Centroids", zorder=10)
        plt.legend()
        plt.show()

    numberOfRows, numberOfColumns = X.shape
    sse = 0.0  # Τετραγωνικό σφάλμα
    for i in range(k):  # Για κάθε συστάδα
        for j in range(numberOfRows):
            if IDX[j] == i:
                sse = sse + math.dist(X[j], C[i]) ** 2  # Υπολόγισε την Ευκλείδεια απόσταση
    print(f"Sum of Squared Errors for columns {idx_0, idx_1}: {sse}")

# Το τρίτο όρισμα είναι μία σημαία για plot

classification(0, 1, True)  # Δύο πρώτες στήλες
classification(296, 305)
classification(467, 468) # Δύο τελευταίες στήλες
classification(205, 175)