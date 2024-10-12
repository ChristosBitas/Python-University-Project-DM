from sklearn.datasets import load_iris
meas = load_iris().data
from sklearn.cluster import KMeans
import math
from sklearn import metrics

# Χρησιμοποιούνται οι 2 τελευταίες διαστάσεις του πίνακα
X = meas[:, [2, 3]]
k = 3 # Oρίζεται ότι τα δεδομένα θα οργανωθούν σε 3 συστάδες
kmeans = KMeans(n_clusters=k).fit(X) # Εφαρμογή του k‐means
IDX = kmeans.labels_
C = kmeans.cluster_centers_
import matplotlib.pyplot as plt
plt.figure(1)
# Παρουσιάζεται η κλάση που ανήκει η κάθε παρατήρηση
plt.plot(IDX[:],'o')
plt.show()
plt.plot(X[IDX==0][:,0], X[IDX==0][:,1], 'limegreen', marker='o', linewidth=0, label='C1')
plt.plot(X[IDX==1][:,0], X[IDX==1][:,1], 'yellow', marker='o', linewidth=0, label='C2')
plt.plot(X[IDX==2][:,0], X[IDX==2][:,1], 'c.', marker='o', label='C3')
plt.scatter(C[:,0], C[:,1], marker='x', color='black', s=150 , linewidth=3, label="Centroids", zorder=10)
plt.legend()
plt.show()

# ΕΥΡΕΣΗ SSE

k_table = range(2, 10)
numberOfRows, numberOfColumns = X.shape

sse_table = []
silhouette_table = []
for k in k_table:  # Για κάθε k
    kmeans = KMeans(n_clusters=k)  # Αρχικοποίηση μοντέλου
    kmeans.fit(X)  # εκπαίδευση μοντέλου
    IDX = kmeans.labels_  # ετικέτες
    C = kmeans.cluster_centers_  # κεντροειδή
    sse = 0.0  # Τετραγωνικό σφάλμα
    for i in range(k):  # Για κάθε συστάδα
        for j in range(numberOfRows):
            if IDX[j] == i:
                sse = sse + math.dist(X[j], C[i]) ** 2  # Υπολόγισε την Ευκλείδεια απόσταση
    sse_table.append(sse)  # Πρόσθεσε το στον πίνακα
    silhouette_table.append(metrics.silhouette_score(X, IDX))

for k, sil in zip(k_table, silhouette_table):
    print(f"k: {k} || Silhouette Coefficient: %0.3f" %sil)

plt.plot(k_table, sse_table)  # Παρουσίαση
plt.xlabel('K-Value')
plt.ylabel('Sum of Squared Errors')
plt.show()

plt.plot(k_table, silhouette_table)  # Παρουσίαση
plt.xlabel('K-Value')
plt.ylabel('Silhouette Coefficient')
plt.show()