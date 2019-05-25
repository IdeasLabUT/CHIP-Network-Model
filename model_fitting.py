import time
import numpy as np
import dataset_utils
import generative_model_utils as utils
import community_model_fitting_utils as model_utils

dataset = 'fb'
# dataset = 'enron'

plot_fitted_hist = True
verbose = False
k_values_to_test = list(range(1, 11))

# Loading data
if dataset == 'fb':
    ((train_event_dict, train_num_nodes, train_duration),
     (test_event_dict, test_num_nodes, test_duration),
     (combined_event_dict, combined_num_nodes, combined_duration),
     nodes_not_in_train) = dataset_utils.load_fb_train_test()

else:
    ((train_event_dict, train_num_nodes, train_duration),
     (test_event_dict, test_num_nodes, test_duration),
     (combined_event_dict, combined_num_nodes, combined_duration),
     nodes_not_in_train) = dataset_utils.load_enron_train_test()

total_tic = time.clock()
print(f"{dataset}'s Log-likelihoods per event:")

lls_per_event = []
for num_classes in k_values_to_test:
    if verbose:
        print("K:", num_classes)

    tic = time.clock()

    # Fitting the model to the train data
    train_node_membership, train_bp_mu, train_bp_alpha, train_bp_beta, train_block_pair_events = \
        model_utils.fit_community_model(train_event_dict, train_num_nodes, train_duration, num_classes, verbose=verbose)

    # Add nodes that were not in train to the largest block
    combined_node_membership = model_utils.assign_node_membership_for_missing_nodes(train_node_membership,
                                                                                    nodes_not_in_train)
    # Calculate log-likelihood given the entire dataset
    combined_block_pair_events = utils.event_dict_to_block_pair_events(combined_event_dict, combined_node_membership,
                                                                       num_classes)

    combined_log_likelihood = model_utils.calc_full_log_likelihood(combined_block_pair_events, combined_node_membership,
                                                                   train_bp_mu, train_bp_alpha, train_bp_beta,
                                                                   combined_duration, num_classes)

    # Calculate log-likelihood given the train dataset
    train_log_likelihood = model_utils.calc_full_log_likelihood(train_block_pair_events, train_node_membership,
                                                                train_bp_mu, train_bp_alpha, train_bp_beta,
                                                                train_duration, num_classes)

    if verbose:
        train_n_events = np.sum(utils.event_dict_to_aggregated_adjacency(train_num_nodes, train_event_dict))
        print(f"Train ll: {train_log_likelihood / train_n_events:.4f}")

    # Calculate per event log likelihood
    ll_per_event = model_utils.calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood,
                                                             test_event_dict, test_num_nodes)

    toc = time.clock()
    lls_per_event.append(ll_per_event)

    print(f"K: {num_classes} - Test ll: {ll_per_event:.3f} - Took: {toc - tic:.2f}s")

    if plot_fitted_hist:
        model_utils.plot_real_vs_fitted_count_histogram(train_event_dict, train_node_membership,
                                                        train_bp_mu, train_bp_alpha, train_bp_beta,
                                                        train_duration)

total_toc = time.clock()

print(f"Total time elapsed: {total_toc - total_tic:.2f}s")
