# -*- coding: utf-8 -*-
"""
@author: Makan Arastuie
"""

import time
import pickle
import numpy as np
import dataset_utils
import generative_model_utils as utils
import model_fitting_utils as model_utils
import bhm_parameter_estimation as estimate_utils


def fit_and_eval_block_hawkes(train_tuple, test_tuple, combined_tuple, nodes_not_in_train,
                              k_values_to_test=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                              local_search_max_iter=0, local_search_n_cores=-1,
                              plot_fitted_hist=False, verbose=False):

    """
    Fits the Block Hawkes model (BHM) to train and evaluates the log-likelihood on the test, by evaluating the
    log-likelihood on the combined dataset and subtracting the likelihood of train, dividing by number of events in test

    :param train_tuple, test_tuple, combined_tuple: A tuple of (event dict, number of nodes, duration)
    :param nodes_not_in_train: Nodes that are in the test data, but not in the train
    :param k_values_to_test: iterable obj of number of communities to fit
    :param local_search_max_iter: if >0, then the model is fitted using local search, else local search is not used.
    :param local_search_n_cores: Number of cores to be used for local search. Ignored if local_search_max_iter <= 0.
    :param plot_fitted_hist: If True, plots a histogram of the event count of read vs. fitted model.
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
        train_node_membership, train_bp_mu, train_bp_alpha, train_bp_beta, train_block_pair_events = \
            estimate_utils.fit_block_model(train_event_dict, train_num_nodes, train_duration, num_classes,
                                           local_search_max_iter, local_search_n_cores,
                                           verbose=verbose)

        # Add nodes that were not in train to the largest block
        combined_node_membership = model_utils.assign_node_membership_for_missing_nodes(train_node_membership,
                                                                                        nodes_not_in_train)

        # Calculate log-likelihood given the entire dataset
        combined_block_pair_events = estimate_utils.event_dict_to_combined_block_pair_events(combined_event_dict,
                                                                                             combined_node_membership,
                                                                                             num_classes)

        combined_log_likelihood = estimate_utils.calc_full_log_likelihood(combined_block_pair_events,
                                                                          combined_node_membership,
                                                                          train_bp_mu, train_bp_alpha, train_bp_beta,
                                                                          combined_duration, num_classes,
                                                                          add_com_assig_log_prob=True)

        # Calculate log-likelihood given the train dataset
        train_log_likelihood = estimate_utils.calc_full_log_likelihood(train_block_pair_events, train_node_membership,
                                                                       train_bp_mu, train_bp_alpha, train_bp_beta,
                                                                       train_duration, num_classes,
                                                                       add_com_assig_log_prob=True)

        # Calculate per event log likelihood
        ll_per_event = model_utils.calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood,
                                                                 test_event_dict, test_num_nodes)

        toc = time.time()
        lls_per_event.append(ll_per_event)

        # Print train and test log-likelihood per event
        train_n_events = np.sum(utils.event_dict_to_aggregated_adjacency(train_num_nodes, train_event_dict))
        print(f"K: {num_classes} - Train ll: {train_log_likelihood / train_n_events:.4f}", end=' - ')
        print(f"Test ll: {ll_per_event:.3f} - Took: {toc - tic:.2f}s")

        # Save results
        result_file_path = '/shared/Results/CommunityHawkes/fb'
        with open(f'{result_file_path}/k{num_classes}-model-params.pckl', 'wb') as handle:
            pickle.dump([train_node_membership, train_bp_mu, train_bp_alpha, train_bp_beta, train_block_pair_events],
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        if plot_fitted_hist:
            estimate_utils.generate_fit_block_hawkes(train_event_dict, train_node_membership,
                                                     train_bp_mu, train_bp_alpha, train_bp_beta,
                                                     train_duration)

    total_toc = time.time()

    print(f"Total time elapsed: {total_toc - total_tic:.2f}s")

    return lls_per_event


# Running Block Hawkes model on Facebook, Enron, Reality Mining, and simulated data
if __name__ == "__main__":
    # Entire Facebook Dataset
    print("Facebook wall-post dataset")
    fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train = \
        dataset_utils.load_facebook_wall(timestamp_max=1000, largest_connected_component_only=True, train_percentage=0.8)
    fit_and_eval_block_hawkes(fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train,
                              local_search_max_iter=500, local_search_n_cores=25,
                              k_values_to_test=[1],
                              plot_fitted_hist=False, verbose=False)

    # # Facebook Dataset
    # print("Facebook wall-post dataset")
    # fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train = \
    #     dataset_utils.load_fb_train_test(remove_nodes_not_in_train=True)
    # fit_and_eval_block_hawkes(fb_train_tuple, fb_test_tuple, fb_combined_tuple, fb_nodes_not_in_train,
    #                           local_search_max_iter=500, local_search_n_cores=25,
    #                           k_values_to_test=[1, 2, 3],
    #                           plot_fitted_hist=False, verbose=False)

    # # Enron Dataset
    # print("Enron dataset")
    # k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
    #            90, 95, 100]
    # enron_train_tuple, enron_test_tuple, enron_combined_tuple, enron_nodes_not_in_train = \
    #     dataset_utils.load_enron_train_test(remove_nodes_not_in_train=False)
    # fit_and_eval_block_hawkes(enron_train_tuple, enron_test_tuple, enron_combined_tuple, enron_nodes_not_in_train,
    #                           local_search_max_iter=100000, local_search_n_cores=6,
    #                           k_values_to_test=[enron_train_tuple[1]], plot_fitted_hist=False, verbose=False)

    # # # Reality Mining
    # # print("Reality Mining")
    # rm_train_tuple, rm_test_tuple, rm_combined_tuple, rm_nodes_not_in_train = \
    #     dataset_utils.load_reality_mining_test_train(remove_nodes_not_in_train=True)
    # fit_and_eval_block_hawkes(rm_train_tuple, rm_test_tuple, rm_combined_tuple, rm_nodes_not_in_train,
    #                           local_search_max_iter=100000, local_search_n_cores=26,
    #                           k_values_to_test=[rm_train_tuple[1]], plot_fitted_hist=False, verbose=False)

    # # Simulated Data
    # print("Simulated Data:")
    # seed = None
    # n_classes = 4
    # n_nodes = 64
    # duration = 50
    # class_probs = np.ones(n_classes) / n_classes
    #
    # alpha = 0.6
    # beta = 0.8
    # mu_diag = 1.6
    # mu_off_diag = 0.8
    #
    # bp_alpha = np.ones((n_classes, n_classes), dtype=np.float) * alpha
    # bp_beta = np.ones((n_classes, n_classes), dtype=np.float) * beta
    # bp_mu = np.ones((n_classes, n_classes), dtype=np.float) * mu_off_diag
    # np.fill_diagonal(bp_mu, mu_diag)
    #
    # sim_node_membership, sim_event_dict = block_generative_model(n_nodes, class_probs,
    #                                                             bp_mu, bp_alpha, bp_beta,
    #                                                             duration, seed=seed)
    #
    # sim_event_list = utils.event_dict_to_event_list(sim_event_dict)
    # sim_train_tuple, sim_test_tuple, sim_combined_tuple, sim_nodes_not_in_train = \
    #     dataset_utils.split_event_list_to_train_test(sim_event_list, train_percentage=0.8)
    #
    # print(sim_train_tuple[-1])
    # print(sim_test_tuple[-1])
    # print(sim_combined_tuple[-1])
    # fit_and_eval_block_hawkes(sim_train_tuple, sim_test_tuple, sim_combined_tuple, sim_nodes_not_in_train)