# _*_ coding: utf-8 _*_
"""
Time:     2022/2/3 11:48
Author:   ChenXin
Version:  V 0.1
File:     DBSCANCluster.py
Describe:  Github link: https://github.com/Chen-X666
"""
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

def DBSCANCluster_plt(data):
    plt.figure(figsize=(10, 5))
    nn = NearestNeighbors(n_neighbors=1000).fit(data)
    distances, idx = nn.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()

def DBSCANCluster(data):
    db = DBSCAN(eps=0.2, min_samples=2000).fit(data)
    labels = db.labels_  # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))
    return labels


