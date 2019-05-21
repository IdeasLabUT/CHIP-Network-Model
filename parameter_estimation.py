# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:34:50 2019

@author: kevin
"""

import numpy as np
import tick.hawkes as tick
import generative_model_utils as utils
from community_generative_model import community_generative_model

from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti, HawkesExpKern
import matplotlib.pyplot as plt


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

    block_pair_events = utils.event_dict_to_block_pair_events(event_dicts, class_assignment, n_classes)

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


if __name__ == "__main__":
    seed = None
    number_of_nodes = 64
    class_probabilities = [0.25, 0.25, 0.25, 0.25]
    num_of_classes = len(class_probabilities)
    end_time = 10000
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

    mu_estimates, alpha_estimates = estimate_hawkes_kernel(event_dicts, node_membership, num_of_classes, bp_beta)

    for b_i in range(num_of_classes):
        for b_j in range(num_of_classes):
            print("Block pair", b_i, b_j)
            print("Alpha: True:", bp_alpha[b_i, b_j], "Estimate:", alpha_estimates[b_i, b_j])
            print("Mu: True:", bp_mu[b_i, b_j], "Estimate:", mu_estimates[b_i, b_j])
