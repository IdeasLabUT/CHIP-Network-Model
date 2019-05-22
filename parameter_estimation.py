# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:34:50 2019

@author: kevin
"""

import numpy as np
import tick.hawkes as tick
import matplotlib.pyplot as plt
import generative_model_utils as utils
from community_generative_model import community_generative_model
from scipy.optimize import minimize_scalar


def estimate_hawkes_from_counts(agg_adj, class_vec, duration):
    num_classes = class_vec.max()+1
    sample_mean = np.zeros((num_classes,num_classes))
    sample_var = np.zeros((num_classes,num_classes))
    for a in range(num_classes):
        for b in range(num_classes):
            nodes_in_a = np.where(class_vec==a)[0]
            nodes_in_b = np.where(class_vec==b)[0]
            agg_adj_block = agg_adj[nodes_in_a[:,np.newaxis],nodes_in_b]
            if a == b:
                # For diagonal blocks, need to make sure we're not including
                # the diagonal entries of the adjacency matrix in our
                # calculations, so extract indices for the lower and upper
                # triangular portions
                num_nodes_in_a = nodes_in_a.size
                lower_indices = np.tril_indices(num_nodes_in_a,-1)
                upper_indices = np.triu_indices(num_nodes_in_a,1)
                agg_adj_block_no_diag = np.r_[agg_adj_block[lower_indices],
                                              agg_adj_block[upper_indices]]
                sample_mean[a,b] = np.mean(agg_adj_block_no_diag)
                sample_var[a,b] = np.var(agg_adj_block_no_diag,ddof=1)
                
            else:
                sample_mean[a,b] = np.mean(agg_adj_block)
                sample_var[a,b] = np.var(agg_adj_block,ddof=1)
    
    mu = np.sqrt(sample_mean**3 / sample_var) / duration
    alpha_beta_ratio = 1 - np.sqrt(sample_mean / sample_var)
    return (mu, alpha_beta_ratio)


def estimate_hawkes_kernel(event_dicts, class_assignment, n_classes, bp_beta, learner_param_dict=None):
    """
    Estimates mu and alpha for a network given the event_dict and a fixed and given beta/decay.

    :param event_dicts: Edge dictionary of events between all node pair. Output of the generative models.
    :param class_assignment: membership of every node to one of K classes. num_nodes x 1 (class of node i)
    :param n_classes: (int) number of classes
    :param bp_beta: K x K matrix where entry ij denotes the beta/decay of Hawkes process for block pair (b_i, b_j)
    :param learner_param_dict: dict of parameters for tick's hawkes kernel. If `None` default values will be used.
                                Check tick's `HawkesExpKern` for parameters. Check `default_param` for defaults.
    :return: `mu_estimate` and `alpha_estimates` both K x K matrices where entry ij denotes the estimated mu and alpha
             of the Hawkes process for block pair (b_i, b_j).
    """
    # Setting up parameters for estimation
    default_params = {'penalty': 'l2',
                      'C': 0.1,
                      'gofit': 'least-squares',
                      'verbose': True,
                      'tol': 1e-11,
                      'solver': 'gd',
                      'step': 1e-3,
                      'max_iter': 1000}

    if learner_param_dict is not None:
        default_params.update(learner_param_dict)

    block_pair_events = utils.event_dict_to_block_pair_events(event_dicts, class_assignment, n_classes,
                                                              is_for_tick=True)

    alpha_estimates = np.zeros((n_classes, n_classes))
    mu_estimates = np.zeros((n_classes, n_classes))

    for b_i in range(n_classes):
        for b_j in range(n_classes):
            learner = tick.HawkesExpKern(bp_beta[b_i, b_j], penalty=default_params['penalty'], C=default_params['C'],
                                         gofit=default_params['gofit'], verbose=default_params['verbose'],
                                         tol=default_params['tol'], solver=default_params['solver'],
                                         step=default_params['step'], max_iter=default_params['max_iter'])

            learner.fit(block_pair_events[b_i][b_j], start=0.1)

            alpha_estimates[b_i, b_j] = learner.adjacency[0][0] / bp_beta[b_i, b_j]
            mu_estimates[b_i, b_j] = learner.baseline[0]

    return mu_estimates, alpha_estimates


def compute_wijs(np_events, beta):
    n_events = len(np_events)
    if n_events <= 1:
        return 0

    wijs = np.zeros((n_events - 1))

    for q in range(1, n_events):
        wijs[q - 1] = np.sum(np.exp(-beta * (np_events[q] - np_events[:q])))

    if n_events == 1:
        print(wijs)
    return wijs


def compute_vijs(np_events, beta):
    n_events = len(np_events)
    if n_events <= 1:
        return 0

    vijs = np.zeros((n_events - 1))

    for q in range(1, n_events):
        q_shifted_events = np_events[q] - np_events[:q]
        vijs[q - 1] = np.sum(q_shifted_events * np.exp(-beta * q_shifted_events))

    return vijs


def full_log_likelihood(bp_events, mu, alpha, beta, end_time):
    ll = 0
    for np_events in bp_events:
        ll += -mu * end_time

        if len(np_events) == 0:
            continue

        second_inner_sum = np.sum((alpha / beta) * (np.exp(-beta * (end_time - np_events)) - 1))
        third_inner_sum = np.sum(np.log(mu + alpha * compute_wijs(np_events, beta)))

        ll += second_inner_sum + third_inner_sum

    return ll

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


def log_likelihood_mu_deriv(bp_events, mu, alpha, beta, end_time):
    ll = 0
    for np_events in bp_events:
        ll += -end_time

        if len(np_events) == 0:
            continue

        ll += np.sum(1 / (mu + alpha * compute_wijs(np_events, beta)))

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
            result.append(full_log_likelihood(bp_events, mu, val, beta, end_time))
            print(val, end='\r')

    elif variable_param == "beta":
        true_val = beta
        for val in values_to_test:
            result.append(full_log_likelihood(bp_events, mu, alpha, val, end_time))
            print(val, end='\r')

    elif variable_param == "mu":
        true_val = mu
        for val in values_to_test:
            result.append(full_log_likelihood(bp_events, val, alpha, beta, end_time))
            print(val, end='\r')

    print()
    plt.plot(values_to_test, result, c='red')
    plt.axvline(x=true_val)
    plt.xlabel(variable_param)
    plt.ylabel(f"Log-likelihood")
    plt.show()

def neg_log_likelihood_beta(beta, bp_events, mu, alpha_beta_ratio, end_time):
    alpha = alpha_beta_ratio*beta
    return -full_log_likelihood(bp_events, mu, alpha, beta, end_time)

def estimate_beta_from_events(bp_events, mu, alpha_beta_ratio, end_time, 
                              tol=1e-3):
    res = minimize_scalar(neg_log_likelihood_beta, method='brent',
                          args=(bp_events, mu, alpha_beta_ratio, end_time))
    return res.x, res

if __name__ == "__main__":
    # Everything below from this point on is only for testing.
    seed = None
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
    # mu_estimates, alpha_estimates = estimate_hawkes_kernel(event_dicts, node_membership, num_of_classes, bp_beta)

    block_pair_events = utils.event_dict_to_block_pair_events(event_dicts, node_membership, num_of_classes)

    for b_i in range(num_of_classes):
        for b_j in range(num_of_classes):
            # Plotting log-likelihood derivatives
            plot_likelihood_deriv("alpha", np.arange(bp_alpha[b_i, b_j] - 5, bp_alpha[b_i, b_j] + 15, 0.2),
                                  block_pair_events[b_i][b_j],
                             bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)

            plot_likelihood_deriv("mu", np.arange(0, bp_mu[b_i, b_j] + .005, 0.0001),
                             block_pair_events[b_i][b_j],
                             bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)

            plot_likelihood_deriv("beta", np.arange(bp_beta[b_i, b_j] - 5, bp_beta[b_i, b_j] + 15, 0.2),
                             block_pair_events[b_i][b_j],
                             bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j], end_time)


            # plotting log-likelihood
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


            # Tick estimation
            # print("Block pair", b_i, b_j)
            # print("Alpha: True:", bp_alpha[b_i, b_j], "Estimate:", alpha_estimates[b_i, b_j])
            # print("Mu: True:", bp_mu[b_i, b_j], "Estimate:", mu_estimates[b_i, b_j])

