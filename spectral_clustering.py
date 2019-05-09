import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


def spectral_cluster(adj, num_classes=2, n_kmeans_init=10):
    # Compute largest num_classes singular values and vectors of adjacency matrix
    u, s, v = svds(adj, k=num_classes)
    v = v.T

    # Sort in decreasing order of magnitude
    sorted_ind = np.argsort(-s)
    u = u[:, sorted_ind]
    v = v[:, sorted_ind]

    z = np.c_[u, v]
    norm_z = normalize(z, norm='l2', axis=1)

    km = KMeans(n_clusters=num_classes, n_init=n_kmeans_init)
    cluster_pred = km.fit_predict(norm_z)

    return cluster_pred
