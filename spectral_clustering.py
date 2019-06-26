import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def spectral_cluster(adj, num_classes=2, n_kmeans_init=10, normalize_z=True, verbose=False, plot_eigenvalues=False):
    # Compute largest num_classes singular values and vectors of adjacency matrix
    u, s, v = svds(adj, k=num_classes)
    v = v.T

    if verbose:
        print("Eigenvalues: \n", s)

    if plot_eigenvalues:
        plt.scatter(np.arange(num_classes, 0, -1), s, marker='*', )
        plt.show()

    # Sort in decreasing order of magnitude
    sorted_ind = np.argsort(-s)
    u = u[:, sorted_ind]
    v = v[:, sorted_ind]

    z = np.c_[u, v]

    if normalize_z:
        z = normalize(z, norm='l2', axis=1)

    km = KMeans(n_clusters=num_classes, n_init=n_kmeans_init)
    cluster_pred = km.fit_predict(z)

    return cluster_pred


def regularized_spectral_cluster(adj, num_classes=2, tau=None, n_kmeans_init=10, normalize_z=True):
    node_outdegree = np.sum(adj, axis=1)
    node_indegree = np.sum(adj, axis=0)

    if tau is None:
        tau = np.mean(node_outdegree)

    o_t = np.diag(1 / np.sqrt(node_outdegree + tau))
    p_t = np.diag(1 / np.sqrt(node_indegree + tau))
    l_t = o_t.dot(adj).dot(p_t)

    return spectral_cluster(l_t, num_classes, n_kmeans_init, normalize_z)
