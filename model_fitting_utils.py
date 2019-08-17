# -*- coding: utf-8 -*-
"""
@author: Makan Arastuie
"""

import numpy as np
import chip_local_search as cls
import matplotlib.pyplot as plt
from scipy.stats import multinomial
import generative_model_utils as utils
import parameter_estimation as estimate_utils
from spectral_clustering import spectral_cluster
from chip_generative_model import community_generative_model


def fit_community_model(event_dict, num_nodes, duration, num_classes, local_search_max_iter, local_search_n_cores,
                        verbose=False):
    """
    Fits CHIP model to a network.

    :param event_dict: Edge dictionary of events between all node pair.
    :param num_nodes: (int) Total number of nodes
    :param duration: (int) duration of the network
    :param num_classes: (int) number of blocks / classes
    :param local_search_max_iter: Maximum number of local search to be performed. If 0, no local search is done
    :param local_search_n_cores: Number of cores to parallelize local search. Only applicable if
                                 `local_search_max_iter` > 0
    :param verbose: Prints fitted Block Hawkes parameters

    :return: node_membership, mu, alpha, beta, block_pair_events
    """

    agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)
    # adj = utils.event_dict_to_adjacency(num_nodes, event_dict)

    # Running spectral clustering
    node_membership = spectral_cluster(agg_adj, num_classes, verbose=False)

    if local_search_max_iter > 0 and num_classes > 1:
        node_membership, bp_mu, bp_alpha, bp_beta = cls.chip_local_search(event_dict, num_classes, node_membership,
                                                                          duration,
                                                                          max_iter=local_search_max_iter,
                                                                          n_cores=local_search_n_cores,
                                                                          return_fitted_param=True, verbose=False)

        block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, num_classes)

    else:
        bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                                duration,
                                                                                1e-10 / duration)
        bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)

        block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, num_classes)

        for b_i in range(num_classes):
            for b_j in range(num_classes):
                bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
                if b_i == b_j:
                    bp_size -= len(np.where(node_membership == b_i)[0])

                bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
                                                                                bp_mu[b_i, b_j],
                                                                                bp_alpha_beta_ratio[b_i, b_j],
                                                                                duration, bp_size)

        bp_alpha = bp_alpha_beta_ratio * bp_beta

    # Printing information about the fit
    if verbose:
        _, block_count = np.unique(node_membership, return_counts=True)
        class_prob = block_count / sum(block_count)

        print(f"Membership percentage: ", class_prob)

        print("Mu:")
        print(bp_mu)

        print("\nAlpha:")
        print(bp_alpha)

        print("\nBeta:")
        print(bp_beta)

    return node_membership, bp_mu, bp_alpha, bp_beta, block_pair_events


def estimate_bp_hawkes_params(event_dict, node_membership, duration, num_classes):
    """
    Estimate CHIP Hawkes parameters.

    :param event_dict: Edge dictionary of events between all node pair.
    :param node_membership: (list) membership of every node to one of K classes.
    :param duration: (int) duration of the network
    :param num_classes: (int) number of blocks / classes

    :return: parameters of the CHIP model -> mu, alpha, beta, m
    """

    num_nodes = len(node_membership)

    agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)
    bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                            duration,
                                                                            1e-10 / duration)

    bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)
    block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, num_classes)

    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
                                                                            bp_mu[b_i, b_j],
                                                                            bp_alpha_beta_ratio[b_i, b_j],
                                                                            duration, bp_size)

    bp_alpha = bp_alpha_beta_ratio * bp_beta

    return bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio


def calc_full_log_likelihood(block_pair_events, node_membership,
                             bp_mu, bp_alpha, bp_beta,
                             duration, num_classes,
                             add_com_assig_log_prob=True):
    """
    Calculates the full log likelihood of the CHIP model.

    :param block_pair_events: (list) n_classes x n_classes where entry ij is a list of event lists between nodes in
                              block i to nodes in block j.
    :param node_membership: (list) membership of every node to one of K classes.
    :param bp_mu: n_classes x n_classes where entry ij is the mu of the block pair ij
    :param bp_alpha: n_classes x n_classes where entry ij is the alpha of the block pair ij
    :param bp_beta: n_classes x n_classes where entry ij is the beta of the block pair ij
    :param duration: (int) duration of the network
    :param num_classes: (int) number of blocks / classes
    :param add_com_assig_log_prob: if True, adds the likelihood the community assignment to the total log-likelihood.

    :return: log-likelihood of the CHIP model
    """

    log_likelihood = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            log_likelihood += estimate_utils.block_pair_full_hawkes_log_likelihood(block_pair_events[b_i][b_j],
                                                                                   bp_mu[b_i, b_j], bp_alpha[b_i, b_j],
                                                                                   bp_beta[b_i, b_j], duration,
                                                                                   block_pair_size=bp_size)

    if add_com_assig_log_prob:
        # Adding the log probability of the community assignments to the full log likelihood
        n_nodes = len(node_membership)
        _, block_count = np.unique(node_membership, return_counts=True)
        class_prob_mle = block_count / sum(block_count)
        rv_multi = multinomial(n_nodes, class_prob_mle)
        log_prob_community_assignment = rv_multi.logpmf(block_count)

        log_likelihood += log_prob_community_assignment

    return log_likelihood


def assign_node_membership_for_missing_nodes(node_membership, missing_nodes):
    """
    Assigns the missing nodes to the largest community

    :param node_membership: (list) membership of every node (except missing ones) to one of K classes
    :param missing_nodes: (list) nodes to be assigned a community

    :return: node_membership including missing nodes
    """
    class_idx, class_count = np.unique(node_membership, return_counts=True)
    largest_class_idx = class_idx[np.argmax(class_count)]

    combined_node_membership = np.copy(node_membership)

    missing_nodes.sort()
    for n in missing_nodes:
        combined_node_membership = np.insert(combined_node_membership, n, largest_class_idx)

    return combined_node_membership


def calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood, test_event_dict, test_num_nodes):
    """
    Subtracts the log-likelihood of the entire data from the train data and divides by the number of test events

    :param combined_log_likelihood: (float) log-likelihood of the entire data
    :param train_log_likelihood: (float) log-likelihood of the train data
    :param test_event_dict: event_dict of the test data
    :param test_num_nodes: Number of nodes in the test dataset

    :return: per test event log-likelihood
    """
    test_num_events = np.sum(utils.event_dict_to_aggregated_adjacency(test_num_nodes, test_event_dict))
    return (combined_log_likelihood - train_log_likelihood) / test_num_events


def generate_fit_community_hawkes(event_dict, node_membership,
                                  bp_mu, bp_alpha, bp_beta,
                                  duration, plot_hist, n_cores=1,
                                  seed=None):
    """
    Generates a community model and plots its count histogram against the original event_dict. (if plot_hist is True)

    :param event_dict: Edge dictionary of events between all node pair.
    :param node_membership: (list) membership of every node to one of K classes.
    :param bp_mu, bp_alpha, bp_beta: Hawkes process parameters
    :param duration: duration of the network
    :param plot_hist: if True, plots a histogram of the weighted adjacency of real vs. generated model.
    :param n_cores: number of cores to parallelize the generative process
    :param seed: seed for CHIP generative process

    :return: generated_node_membership, generated_event_dict
    """

    # Generating a network
    n_nodes = len(node_membership)

    _, block_count = np.unique(node_membership, return_counts=True)
    class_prob = block_count / sum(block_count)

    generated_node_membership, generated_event_dict = community_generative_model(n_nodes, class_prob,
                                                                                 bp_mu, bp_alpha, bp_beta,
                                                                                 burnin=None, end_time=duration,
                                                                                 n_cores=n_cores, seed=seed)

    if plot_hist:
        generated_agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, generated_event_dict, dtype=np.int)
        generated_deg_count_flattened = np.reshape(generated_agg_adj, (n_nodes * n_nodes))

        agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict, dtype=np.int)
        deg_count_flattened = np.reshape(agg_adj, (n_nodes * n_nodes))

        plt.hist(deg_count_flattened, bins=50, alpha=0.5, label='Real Data', color='blue', density=True)
        plt.hist(generated_deg_count_flattened, bins=50, alpha=0.5, label='Generated Data', color='red', density=True)

        plt.legend(loc='upper right')
        plt.xlabel('Event Count')
        plt.ylabel('Density')
        plt.title(f'Histogram of the Count Matrix Real Vs. Generated CHIP Model Data - K: {len(class_prob)}'
                  f'\n Mean Count -  Real: {np.mean(agg_adj):.3f} - Generated: {np.mean(generated_agg_adj):.3f}')
        plt.yscale("log")
        plt.show()

    return generated_node_membership, generated_event_dict


def log_binning(counter, bin_count=35):
    """
    Based on https://stackoverflow.com/questions/16489655/plotting-log-binned-network-degree-distributions/16490678
    """
    keys = counter[0]
    values = counter[1]

    max_x = np.log10(max(keys))
    max_y = np.log10(max(values))
    max_base = max([max_x, max_y])

    min_x = np.log10(min(keys))

    bins = np.logspace(min_x, max_base, num=bin_count)

    bin_means_y = np.histogram(keys, bins, weights=values)[0] / np.histogram(keys, bins)[0]
    bin_means_x = np.histogram(keys, bins, weights=keys)[0] / np.histogram(keys, bins)[0]

    return bin_means_x, bin_means_y
