# -*- coding: utf-8 -*-
"""
@author: Kevin Xu and Makan Arastuie
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def spectral_cluster(adj, num_classes=2, n_kmeans_init=10, normalize_z=True, verbose=False, plot_eigenvalues=False,
                     plot_save_path=''):
    """
    Runs spectral clustering on weighted or unweighted adjacency matrix


    :param adj: weighted, unweighted or regularized adjacency matrix
    :param num_classes: number of classes for spectral clustering
    :param n_kmeans_init: number of initializations for k-means
    :param normalize_z: If True, vector z is normalized to sum to 1
    :param verbose: if True, prints the eigenvalues
    :param plot_eigenvalues: if True, plots the first `num_classes` singular values
    :param plot_save_path: directory to save the plot

    :return: predicted clustering membership
    """
    # Compute largest num_classes singular values and vectors of adjacency matrix
    u, s, v = svds(adj, k=num_classes)
    v = v.T

    if verbose:
        print("Eigenvalues: \n", s)

    if plot_eigenvalues:
        fig, ax = plt.subplots()
        plt.scatter(np.arange(num_classes, 0, -1), s, s=80, marker='*', color='blue')
        plt.xlabel('Rank', fontsize=24)
        plt.ylabel('Singular Values', fontsize=24)
        plt.grid(True)
        ax.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig(f'{plot_save_path}/singular_values.pdf')
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
    """
    Runs regularized spectral clustering on weighted or unweighted adjacency matrix

    :param adj: weighted, unweighted or regularized adjacency matrix
    :param num_classes: number of classes for spectral clustering
    :param tau: regularization parameter
    :param n_kmeans_init: number of initializations for k-means
    :param normalize_z: If True, vector z is normalized to sum to 1

    :return: predicted clustering membership
    """
    node_outdegree = np.sum(adj, axis=1)
    node_indegree = np.sum(adj, axis=0)

    if tau is None:
        tau = np.mean(node_outdegree)

    o_t = np.diag(1 / np.sqrt(node_outdegree + tau))
    p_t = np.diag(1 / np.sqrt(node_indegree + tau))
    l_t = o_t.dot(adj).dot(p_t)

    return spectral_cluster(l_t, num_classes, n_kmeans_init, normalize_z)
