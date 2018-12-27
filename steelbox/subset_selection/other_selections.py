import numpy as np
import random
import time
import math
from copy import deepcopy
from sklearn.cluster import KMeans
from .knn import score_subset, update_weight

# ----------------- cluster in ( raw_input / embedded class ) space ----------------
def sub_select_cluster(X, Y, n_samples):
    kmeans = KMeans(n_clusters=n_samples)
    kmeans = kmeans.fit(X)
    
    cluster_labels = list(kmeans.predict(X))
    # print (cluster_labels[:100])
    from sklearn.metrics import pairwise_distances_argmin_min
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    counts = [cluster_labels.count(i) for i in range(n_samples)]
    
    X_sub = []
    Y_sub = []
    for i in range(n_samples):
        X_sub.append(X[closest[i]])
        Y_sub.append(Y[closest[i]])
    return np.array(X_sub), np.array(Y_sub), closest

if __name__ == '__main__':

    def test1():
        from data_raw.artificial import gen_data
        X, Y, X_t, Y_t = gen_data(2000)
        W = np.ones(1000)
        X_rsub, Y_rsub = X[:100, :], Y[:100]
        X_sub, Y_sub, _ = sub_select_cluster(X, Y, 100)
        print ("score of rand subset\n", score_subset(X_rsub, Y_rsub, X, Y, W))
        print ("score of cluster subset\n", score_subset(X_sub, Y_sub, X, Y, W))

    test1()
