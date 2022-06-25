# -*- coding: utf-8 -*-
"""
@author: Makan Arastuie
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import generative_model_utils as utils
from scipy.optimize import minimize_scalar, minimize


def compute_sample_mean_and_variance(agg_adj, class_vec):
    """
    Computes CHIP's sample mean (N) and variance (S^2)

    :param agg_adj: sparse weighted adjacency of the network
    :param class_vec: (list) membership of every node to one of K classes.

    :return: N, S^2
    """
    num_classes = class_vec.max() + 1
    sample_mean = np.zeros((num_classes, num_classes))
    sample_var = np.zeros((num_classes, num_classes))

    community_membership = utils.node_membership_to_community_membership(class_vec, num_classes)

    for a in range(num_classes):
        for b in range(num_classes):
            nodes_in_a = community_membership[a]
            nodes_in_b = community_membership[b]

            # if both block sizes = 1, no need to compute params of that block pair, set it to the default.
            if nodes_in_a.size <= 1 and nodes_in_b.size <= 1:
                sample_mean[a, b] = 0
                sample_var[a, b] = 0
                continue

            agg_adj_block = agg_adj.tocsr()[nodes_in_a[:, np.newaxis], nodes_in_b]
            if a == b:
                # For diagonal blocks, need to make sure we're not including the diagonal
                # entries of the adjacency matrix in our calculations

                # compute number of node pairs in block pair(a, a)
                n = len(nodes_in_a) * (len(nodes_in_a) - 1)
                sample_mean[a, b] = agg_adj_block.sum()/n
                adj_squared = agg_adj_block.copy()
                adj_squared.data **= 2
                sample_var[a, b] = n / (n - 1) * (adj_squared.sum()/n - np.square(sample_mean[a, b]))
            else:
                sample_mean[a, b] = agg_adj_block.mean()
                # compute number of node pairs in block pair(a, b)
                n = agg_adj_block.shape[0] * agg_adj_block.shape[1]
                adj_squared = agg_adj_block.copy()
                adj_squared.data **= 2
                sample_var[a, b] = n / (n - 1) * (adj_squared.mean() - np.square(sample_mean[a, b]))

    return sample_mean, sample_var


def estimate_hawkes_from_counts(agg_adj, class_vec, duration, default_mu=None):
    """
    Estimates CHIP's mu and m.

    :param agg_adj: sparse weighted adjacency of the network
    :param class_vec: (list) membership of every node to one of K classes.
    :param duration: duration of the network
    :param default_mu: default mu values for block pairs with sample variance of 0.

    :return: mu, m
    """

    sample_mean, sample_var = compute_sample_mean_and_variance(agg_adj, class_vec)

    # Variance can be zero, resulting in division by zero warnings. Ignore and set a default mu.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.sqrt(sample_mean**3 / sample_var) / duration
        alpha_beta_ratio = 1 - np.sqrt(sample_mean / sample_var)

    # If sample_var is 0, depending on sample mean, mu and the ratio can be nan or inf. Set them to default values.
    if default_mu is not None:
        mu[np.isnan(mu)] = default_mu
        mu[np.isinf(mu)] = default_mu
        alpha_beta_ratio[np.isnan(alpha_beta_ratio)] = 0
        alpha_beta_ratio[np.isinf(alpha_beta_ratio)] = 0

        # If ratio is negative, set it to 0
        alpha_beta_ratio[alpha_beta_ratio < 0] = 0

    return mu, alpha_beta_ratio


def compute_wijs(np_events, beta):
    """
    Computes the recursive portion of the Hawkes log-likelihood, non-recursively.

    :param np_events: event list / list of timestamps of events from a node in block i to a node in block j.
    :param beta: beta of the block pair ij
    :return: recursive sum of the CHIP Hawkes log-likelihood
    """

    n_events = len(np_events)
    if n_events < 1:
        return 0

    wijs = np.zeros(n_events)
    for q in range(1, n_events):
        wijs[q] = np.sum(np.exp(-beta * (np_events[q] - np_events[:q])))

    return wijs


def compute_wijs_recursive(np_events, beta):
    """
    Computes the recursive portion of the Hawkes log-likelihood, recursively (using last computed value, not an actual
    recursive function)

    :param np_events: event list / list of timestamps of events from a node in block i to a node in block j.
    :param beta: beta of the block pair ij

    :return: recursive sum of the CHIP Hawkes log-likelihood
    """

    n_events = len(np_events)
    if n_events < 1:
        return 0

    wijs = np.zeros(n_events)
    for i in range(1, n_events):
        wijs[i] = np.exp(-beta * (np_events[i] - np_events[i - 1])) * (1 + wijs[i - 1])

    return wijs


def block_pair_full_hawkes_log_likelihood(bp_events, mu, alpha, beta, end_time, block_pair_size=None):
    """

    :param bp_events: (list) n_classes x n_classes where entry ij is a list of event lists between nodes in
                          block i to nodes in block j
    :param mu, alpha, beta: Hawkes parameters of the block pair ij
    :param end_time: duration of the network / the last available timestamp
    :param block_pair_size: Size of the block pair. bp_events may not include an entry for node_pairs with no
                        interactions, in that case, we need to add (-mu * end_time) to the likelihood for each
                        missing node pair

    :return: CHIP Hawkes log likelihood of a block pair
    """
    ll = 0
    for np_events in bp_events:
        ll += -mu * end_time

        if len(np_events) == 0:
            continue

        second_inner_sum = (alpha / beta) * np.sum(np.exp(-beta * (end_time - np_events)) - 1)
        third_inner_sum = np.sum(np.log(mu + alpha * compute_wijs_recursive(np_events, beta)))

        ll += second_inner_sum + third_inner_sum

    if block_pair_size is not None:
        num_missing_node_pairs = block_pair_size - len(bp_events)
        ll += num_missing_node_pairs * -mu * end_time

    return ll


def neg_log_likelihood_beta(beta, bp_events, mu, alpha_beta_ratio, end_time, block_pair_size):
    """
    Returns negative log-likelihood of `block_pair_full_hawkes_log_likelihood`, instead of alpha requires m.
    Check `block_pair_full_hawkes_log_likelihood` doc string for parameters.

    :param alpha_beta_ratio: m of the block pair ij
    :return: Negative CHIP Hawkes log-likelihood
    """
    alpha = alpha_beta_ratio*beta
    return -block_pair_full_hawkes_log_likelihood(bp_events, mu, alpha, beta, end_time, block_pair_size)


def estimate_beta_from_events(bp_events, mu, alpha_beta_ratio, end_time, block_pair_size=None, tol=1e-3):
    """
    Uses scipy minimize_scalar to as a line search to find beta.
    Check `block_pair_full_hawkes_log_likelihood` doc string for parameters.

    :param alpha_beta_ratio: m of the block pair ij
    :param tol: tol of minimize_scalar.
    :return: beta and details on minimize_scalar
    """
    res = minimize_scalar(neg_log_likelihood_beta, method='bounded', bounds=(0, 10),
                          args=(bp_events, mu, alpha_beta_ratio, end_time, block_pair_size))
    return res.x, res


def neg_log_likelihood_all(param, bp_events, end_time, block_pair_size=None):
    """
    Returns negative log-likelihood of `block_pair_full_hawkes_log_likelihood`.
    Check `block_pair_full_hawkes_log_likelihood` doc string for parameters.

    :param param: tuple of Hawkes parameters (alpha, beta, mu)
    :return: negative log-likelihood of CHIP Hawkes
    """
    alpha = param[0]
    beta = param[1]
    mu = param[2]
    return -block_pair_full_hawkes_log_likelihood(bp_events, mu, alpha, beta,
                                                  end_time, block_pair_size)


def estimate_all_from_events(bp_events, end_time, init_param=(1e-2,2e-2,2e-5), block_pair_size=None, tol=1e-3):
    """
    Estimates mu, alpha and beta for a single CHIP block-pair using L-BFGS-B.
    Check `block_pair_full_hawkes_log_likelihood` doc string for parameters.


    :param init_param: tuple of initial parameters for the CHIP Hawkes (alpha, beta, mu)
    :param tol: tol of scipy.minimize

    :return: alpha, beta, mu and details of the L-BFGS-B method.
    """
    res = minimize(neg_log_likelihood_all, init_param, method='L-BFGS-B',
                   bounds=((0, None), (0, None), (0, None)), jac=None,
                   args=(bp_events, end_time, block_pair_size))
    return res.x, res
