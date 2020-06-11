# -*- coding: utf-8 -*-
"""
@author: Makan Arastuie
"""

import time
import numpy as np
import dataset_utils
import generative_model_utils as utils
import model_fitting_utils as model_utils
from scipy.stats import multinomial, expon
from spectral_clustering import spectral_cluster


def fit_and_eval_poisson_baseline(train_tuple, test_tuple, combined_tuple, nodes_not_in_train,
                                  k_values_to_test, verbose=False):

    """
    Fits the Poisson baseline model to train and evaluates the log-likelihood on the test, by evaluating the
    log-likelihood on the combined dataset and subtracting the likelihood of train, dividing by number of events in test

    This model is basically like a BHM model, but models interactions as a Poisson. Keep in mind that modeling
    interactions as Poisson makes the BHM model the same as CHIP in terms of likelihood, since generating events at
    the node-pair level with lambda * 1/block pair size, is equivalent to generating at the block pair level with
    lambda, then thinning.

    :param train_tuple, test_tuple, combined_tuple: A tuple of (event dict, number of nodes, duration)
    :param nodes_not_in_train: Nodes that are in the test data, but not in the train
    :param k_values_to_test: iterable obj of number of communities to fit
    :param verbose: Prints details of the fit along the way.

    :return: (list) test log-likelihood per event for all `k_values_to_test`.
    """

    train_event_dict, train_num_nodes, train_duration = train_tuple
    test_event_dict, test_num_nodes, test_duration = test_tuple
    combined_event_dict, combined_num_nodes, combined_duration = combined_tuple

    total_tic = time.time()
    print("Log-likelihoods per event:")

    lls_per_event = []
    for num_classes in k_values_to_test:
        if verbose:
            print("K:", num_classes)

        tic = time.time()

        # Fitting the model to the train data
        train_node_membership, train_bp_lambda, train_block_count_matrix = \
            fit_poisson_baseline_model(train_event_dict, train_num_nodes, train_duration, num_classes, verbose=verbose)

        # Add nodes that were not in train to the largest block
        combined_node_membership = model_utils.assign_node_membership_for_missing_nodes(train_node_membership,
                                                                                        nodes_not_in_train)

        # Calculate log-likelihood given the entire dataset
        combined_count_matrix = event_dict_to_block_pair_event_counts(combined_event_dict, combined_node_membership,
                                                                      num_classes)

        combined_log_likelihood = calc_full_log_likelihood(combined_count_matrix, combined_node_membership,
                                                           combined_duration, train_bp_lambda, num_classes)

        # Calculate log-likelihood given the train dataset
        train_log_likelihood = calc_full_log_likelihood(train_block_count_matrix, train_node_membership,
                                                        test_duration, train_bp_lambda, num_classes)

        # Calculate per event log likelihood
        ll_per_event = model_utils.calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood,
                                                                 test_event_dict, test_num_nodes)

        toc = time.time()
        lls_per_event.append(ll_per_event)

        # Print train and test log-likelihood per event
        train_n_events = np.sum(utils.event_dict_to_aggregated_adjacency(train_num_nodes, train_event_dict))
        print(f"K: {num_classes} - Train ll: {train_log_likelihood / train_n_events:.4f}", end=' - ')
        print(f"Test ll: {ll_per_event:.3f} - Took: {toc - tic:.2f}s")

    total_toc = time.time()

    print(f"Total time elapsed: {total_toc - total_tic:.2f}s")

    return lls_per_event


def fit_poisson_baseline_model(event_dict, num_nodes, duration, num_classes, verbose=False):
    """
    Fits a Poisson baseline model to a network.

    :param event_dict: Edge dictionary of events between all node pair.
    :param num_nodes: (int) Total number of nodes
    :param duration: (int) duration of the network
    :param num_classes: (int) number of blocks / classes
    :param verbose: Prints fitted Poisson baseline parameters

    :return: node_membership, lambda, block_pair_events
    """
    # adj = utils.event_dict_to_adjacency(num_nodes, event_dict)
    agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)

    # if number of there are as many classes as nodes, assign each node to its own class
    if num_classes == num_nodes:
        node_membership = list(range(num_nodes))
    else:
        # Running spectral clustering
        node_membership = spectral_cluster(agg_adj, num_classes=num_classes)

    count_matrix = event_dict_to_block_pair_event_counts(event_dict, node_membership, num_classes)

    bp_lambda = estimate_poisson_lambda(count_matrix, node_membership, duration, num_classes,
                                        default_lambda=1e-10 / duration)

    # Printing information about the fit
    if verbose:
        _, block_count = np.unique(node_membership, return_counts=True)
        class_prob = block_count / sum(block_count)

        print(f"Membership percentage: ", class_prob)

        print("Lambda:")
        print(bp_lambda)

    return node_membership, bp_lambda, count_matrix


def event_dict_to_block_pair_event_counts(event_dict, class_assignment, n_classes):
    """
    Converts event_dicts to count matrix of block-pair event counts.

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.
    :param class_assignment: membership of every node to one of K classes. num_nodes x 1 (class of node i)
    :param n_classes: (int) total number of classes
    :return: (np array) n_classes x n_classes where entry ij is denotes the number of events in block-pair ij
    """
    count_matrix = np.zeros((n_classes, n_classes))
    for u, v in event_dict:
        count_matrix[class_assignment[u], class_assignment[v]] += len(event_dict[(u, v)])

    return count_matrix


def estimate_poisson_lambda(count_matrix, node_membership, duration, num_classes, default_lambda=1e-10):
    """
    Estimate lambda for all block pairs.

    :param count_matrix: n_classes x n_classes where entry ij is denotes the number of events in block-pair ij
    :param node_membership: (list) membership of every node to one of K classes.
    :param duration: (int) duration of the network
    :param default_lambda: default value for lambda if there are no events in a block pair to estimate lambda
    :return: n_classes x n_classes where entry ij is the lambda of the block pair ij
    """
    bp_size = utils.calc_block_pair_size(node_membership, num_classes)
    # if a block only has 1 node in it, its own bp_size will be 0.
    # But since count_matrix will be zero setting to 1 won't change the outcome.
    bp_size[bp_size == 0] = 1
    bp_lambda = count_matrix / (duration * bp_size)
    bp_lambda[bp_lambda == 0] = default_lambda

    return bp_lambda


def calc_full_log_likelihood(count_matrix, node_membership, duration, bp_lambda, num_classes,
                             add_com_assig_log_prob=True):
    """
    Calculates the full log likelihood of the Poisson baseline model.

    :param count_matrix: n_classes x n_classes where entry ij is denotes the number of events in block-pair ij
    :param node_membership: (list) membership of every node to one of K classes
    :param duration: (int) duration of the network
    :param bp_lambda: n_classes x n_classes where entry ij is the lambda of the block pair ij
    :param num_classes: (int) number of blocks / classes
    :param add_com_assig_log_prob: if True, adds the likelihood the community assignment to the total log-likelihood.

    :return: log-likelihood of the Poisson baseline model
    """
    log_likelihood = 0

    bp_size = utils.calc_block_pair_size(node_membership, num_classes)
    bp_ll = count_matrix * np.log(bp_lambda) - (bp_lambda * duration * bp_size)
    log_likelihood += np.sum(bp_ll)

    if add_com_assig_log_prob:
        # Adding the log probability of the community assignments to the full log likelihood
        n_nodes = len(node_membership)
        _, block_count = np.unique(node_membership, return_counts=True)
        class_prob_mle = block_count / sum(block_count)
        rv_multi = multinomial(n_nodes, class_prob_mle)
        log_prob_community_assignment = rv_multi.logpmf(block_count)

        log_likelihood += log_prob_community_assignment

    return log_likelihood


# Running Poisson baseline model on Facebook, Enron, Reality Mining
if __name__ == "__main__":
    # Entire Facebook Dataset
    print("Facebook wall-post dataset")
    fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train = \
        dataset_utils.load_facebook_wall(timestamp_max=1000, largest_connected_component_only=True,
                                         train_percentage=0.8)
    fit_and_eval_poisson_baseline(fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train,
                                  k_values_to_test=np.arange(1, 11), verbose=False)

    # # Facebook Dataset
    # print("Facebook wall-post dataset")
    # fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train = \
    #     dataset_utils.load_fb_train_test(remove_nodes_not_in_train=False)
    # fit_and_eval_poisson_baseline(fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train,
    #                               k_values_to_test=np.arange(1, 150), verbose=False)


    # # Enron Dataset
    # print("Enron dataset")
    # enron_train_tuple, enron_test_tuple, enron_combined_tuple, enron_nodes_not_in_train = \
    #     dataset_utils.load_enron_train_test(remove_nodes_not_in_train=False)
    # fit_and_eval_poisson_baseline(enron_train_tuple, enron_test_tuple, enron_combined_tuple, enron_nodes_not_in_train,
    #                               k_values_to_test=np.arange(1, enron_train_tuple[1] + 1),
    #                               verbose=False)
    #
    # # Reality Mining
    # print("Reality Mining")
    # rm_train_tuple, rm_test_tuple, rm_combined_tuple, rm_nodes_not_in_train = \
    #     dataset_utils.load_reality_mining_test_train(remove_nodes_not_in_train=False)
    # fit_and_eval_poisson_baseline(rm_train_tuple, rm_test_tuple, rm_combined_tuple, rm_nodes_not_in_train,
    #                               k_values_to_test=np.arange(1, rm_train_tuple[1] + 1), verbose=False)
