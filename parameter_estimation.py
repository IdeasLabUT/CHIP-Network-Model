# -*- coding: utf-8 -*-
"""
@authors: kevin Xu and Makan Arastuie
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import generative_model_utils as utils
from chip_generative_model import community_generative_model
from scipy.optimize import minimize_scalar, minimize


def compute_sample_mean_and_variance(agg_adj, class_vec):
    """
    Computes CHIP's sample mean (N) and variance (S^2)

    :param agg_adj: weighted adjacency of the network
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

            agg_adj_block = agg_adj[nodes_in_a[:, np.newaxis], nodes_in_b]
            if a == b:
                # For diagonal blocks, need to make sure we're not including
                # the diagonal entries of the adjacency matrix in our
                # calculations, so extract indices for the lower and upper
                # triangular portions

                num_nodes_in_a = nodes_in_a.size
                lower_indices = np.tril_indices(num_nodes_in_a, -1)
                upper_indices = np.triu_indices(num_nodes_in_a, 1)
                agg_adj_block_no_diag = np.r_[agg_adj_block[lower_indices],
                                              agg_adj_block[upper_indices]]

                sample_mean[a, b] = np.mean(agg_adj_block_no_diag)
                sample_var[a, b] = np.var(agg_adj_block_no_diag, ddof=1)

            else:

                sample_mean[a, b] = np.mean(agg_adj_block)
                sample_var[a, b] = np.var(agg_adj_block, ddof=1)

    return sample_mean, sample_var


def estimate_hawkes_from_counts(agg_adj, class_vec, duration, default_mu=None):
    """
    Estimates CHIP's mu and m.

    :param agg_adj: weighted adjacency of the network
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


##### Delete everything below this point in Github master
def compute_vijs(np_events, beta):
    n_events = len(np_events)
    if n_events < 1:
        return 0

    vijs = np.zeros(n_events)
    for q in range(1, n_events):
        q_shifted_events = np_events[q] - np_events[:q]
        vijs[q] = np.sum(q_shifted_events * np.exp(-beta * q_shifted_events))

    return vijs


def neg_log_likelihood_deriv_all(param, bp_events, end_time, block_pair_size=None):
    alpha = param[0]
    beta = param[1]
    mu = param[2]
    return -np.r_[log_likelihood_alpha_deriv(bp_events, mu, alpha, beta, end_time),
             log_likelihood_beta_deriv(bp_events, mu, alpha, beta, end_time),
             log_likelihood_mu_deriv(bp_events, mu, alpha, beta, end_time, block_pair_size)]


def log_likelihood_alpha_deriv(bp_events, mu, alpha, beta, end_time):
    ll = 0
    for np_events in bp_events:
        if len(np_events) == 0:
            continue

        first_inner_sum = np.sum(np.exp(-beta * (end_time - np_events)) - 1) / beta

        wijs = compute_wijs(np_events, beta)
        second_inner_sum = np.sum(wijs / (mu + alpha * wijs))

        ll += first_inner_sum + second_inner_sum

    return ll


def log_likelihood_mu_deriv(bp_events, mu, alpha, beta, end_time, block_pair_size=None):
    ll = 0
    for np_events in bp_events:
        ll += -end_time

        if len(np_events) == 0:
            continue

        ll += np.sum(1 / (mu + alpha * compute_wijs(np_events, beta)))

    if block_pair_size is not None:

        num_missing_node_pairs = block_pair_size - len(bp_events)
        print("deriv", num_missing_node_pairs)
        ll += num_missing_node_pairs * -end_time

    return ll


def log_likelihood_beta_deriv(bp_events, mu, alpha, beta, end_time):
    ll = 0
    for np_events in bp_events:
        if len(np_events) == 0:
            continue

        np_events_t_shifted = end_time - np_events
        first_inner_sum = - alpha * np.sum((1 / beta) * np_events_t_shifted * np.exp(-beta * np_events_t_shifted) +
                                           (1 / beta ** 2) * (np.exp(-beta * np_events_t_shifted) - 1))

        vijs = compute_vijs(np_events, beta)
        wijs = compute_wijs(np_events, beta)
        second_inner_sum = np.sum((alpha * vijs) / (mu + alpha * wijs))

        ll += first_inner_sum - second_inner_sum

    return ll


def plot_likelihood_deriv(deriv, values_to_test, bp_events, mu, alpha, beta, end_time):
    """
    Plots the log-likelihoods to make sure they are maximized at the true parameter.

    :param deriv: (string) "alpha", "beta", "mu"
    :param values_to_test: (iterable) range of values for the derivative parameter.
    :param bp_events: (list) list of lists of events of a single block pair.
    :param mu: True mu/baseline for the block pair.
    :param alpha: True alpha for the block pair.
    :param beta: True beta for the block pair.
    :param end_time: End-time/duration of the hawkes processes.
    """

    result = []
    true_val = 0

    if deriv == "alpha":
        true_val = alpha
        for val in values_to_test:
            result.append(log_likelihood_alpha_deriv(bp_events, mu, val, beta, end_time))
            print(val, end='\r')

    elif deriv == "beta":
        true_val = beta
        for val in values_to_test:
            result.append(log_likelihood_beta_deriv(bp_events, mu, alpha, val, end_time))
            print(val, end='\r')

    elif deriv == "mu":
        true_val = mu
        for val in values_to_test:
            result.append(log_likelihood_mu_deriv(bp_events, val, alpha, beta, end_time))
            print(val, end='\r')

    print()
    plt.plot(values_to_test, result, c='red')
    plt.axvline(x=true_val)
    plt.xlabel(deriv)
    plt.ylabel(f"Log-likelihood's derivative wrt {deriv}")
    plt.show()


def plot_likelihood(variable_param, values_to_test, bp_events, mu, alpha, beta, end_time):
    """
    Plots the log-likelihood to make sure they are maximized at the true parameter, while keeping all parameters except
    `variable_param` constant.

    :param variable_param: (string) "alpha", "beta", "mu". The parameter to vary to calculate log likelihood.
    :param values_to_test: (iterable) range of values for the variable_param parameter.
    :param bp_events: (list) list of lists of events of a single block pair.
    :param mu: True mu/baseline for the block pair.
    :param alpha: True alpha for the block pair.
    :param beta: True beta for the block pair.
    :param end_time: End-time/duration of the hawkes processes.
    """

    result = []
    true_val = 0

    if variable_param == "alpha":
        true_val = alpha
        for val in values_to_test:
            result.append(block_pair_full_hawkes_log_likelihood(bp_events, mu, val, beta, end_time))
            print(val, end='\r')

    elif variable_param == "beta":
        true_val = beta
        for val in values_to_test:
            result.append(block_pair_full_hawkes_log_likelihood(bp_events, mu, alpha, val, end_time))
            print(val, end='\r')

    elif variable_param == "mu":
        true_val = mu
        for val in values_to_test:
            result.append(block_pair_full_hawkes_log_likelihood(bp_events, val, alpha, beta, end_time))
            print(val, end='\r')

    print()
    plt.plot(values_to_test, result, c='red')
    plt.axvline(x=true_val)
    plt.xlabel(variable_param)
    plt.ylabel(f"Log-likelihood")
    plt.show()


def plot_likelihood_scaling_alpha_beta(scalars, bp_events, mu, alpha, beta, end_time):
    """
    Plots the log-likelihood to make sure they are maximized at the true parameter, while keeping all parameters except
    `variable_param` constant.

    :param scalars: (iterator) scales to scale both alpha and beta with, keeping their ratio constant.
    :param bp_events: (list) list of lists of events of a single block pair.
    :param mu: True mu/baseline for the block pair.
    :param alpha: True alpha for the block pair.
    :param beta: True beta for the block pair.
    :param end_time: End-time/duration of the hawkes processes.
    """

    result = []

    for s in scalars:
        result.append(block_pair_full_hawkes_log_likelihood(bp_events, mu, alpha * s, beta * s, end_time))
        print(s, end='\r')

    print()
    plt.plot(scalars, result, c='red')
    plt.title("Log-Likelihood with scaling both alpha and beta")
    plt.xlabel("Scalar")
    plt.axvline(x=1)
    plt.ylabel("Log-likelihood")
    plt.show()


if __name__ == "__main__":
    # Everything below from this point on is only for testing.
    seed = 1
    number_of_nodes = 128
    class_probabilities = [0.25, 0.25, 0.25, 0.25]
    num_of_classes = len(class_probabilities)
    end_time = 100
    burnin = None

    bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 7500
    bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 8000
    bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
    np.fill_diagonal(bp_mu, 1.8)

    bp_mu = utils.scale_parameteres_by_block_pair_size(bp_mu, 128, class_probabilities)
    bp_alpha = utils.scale_parameteres_by_block_pair_size(bp_alpha, 128, class_probabilities)
    bp_beta = utils.scale_parameteres_by_block_pair_size(bp_beta, 128, class_probabilities)

    node_membership, event_dicts = community_generative_model(number_of_nodes,
                                                              class_probabilities,
                                                              bp_mu, bp_alpha, bp_beta,
                                                              burnin, end_time, seed=seed)

    node_membership = utils.one_hot_to_class_assignment(node_membership)

    print("Simulation done.")

    block_pair_events = utils.event_dict_to_block_pair_events(event_dicts, node_membership, num_of_classes)

    for b_i in range(num_of_classes):
        for b_j in range(num_of_classes):
            plot_likelihood_scaling_alpha_beta(np.logspace(np.log10(0.01), np.log10(10), 100), block_pair_events[b_i][b_j],
                                               bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)

            # # Plotting log-likelihood derivatives
            # plot_likelihood_deriv("alpha", np.arange(bp_alpha[b_i, b_j] - 5, bp_alpha[b_i, b_j] + 15, 0.2),
            #                       block_pair_events[b_i][b_j],
            #                       bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)
            #
            # plot_likelihood_deriv("mu", np.arange(0, bp_mu[b_i, b_j] + .005, 0.0001),
            #                       block_pair_events[b_i][b_j],
            #                       bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)
            #
            # plot_likelihood_deriv("beta", np.arange(bp_beta[b_i, b_j] - 5, bp_beta[b_i, b_j] + 15, 0.2),
            #                       block_pair_events[b_i][b_j],
            #                       bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)
            #
            #
            # # plotting log-likelihood
            # plot_likelihood("alpha", np.arange(bp_alpha[b_i, b_j] - 5, bp_alpha[b_i, b_j] + 15, 0.2),
            #                  block_pair_events[b_i][b_j],
            #                  bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)
            #
            # plot_likelihood("mu", np.arange(0, bp_mu[b_i, b_j] + .005, 0.0001),
            #                  block_pair_events[b_i][b_j],
            #                  bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)
            #
            # plot_likelihood("beta", np.arange(bp_beta[b_i, b_j] - 5, bp_beta[b_i, b_j] + 15, 0.2),
            #                  block_pair_events[b_i][b_j],
            #                  bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)

