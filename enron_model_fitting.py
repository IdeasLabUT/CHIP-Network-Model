import numpy as np
import dataset_utils
import generative_model_utils as utils
import parameter_estimation as estimate_utils
import community_model_fitting_utils as model_utils


# Loading data
((train_event_dict, train_num_nodes, train_duration),
 (test_event_dict, test_num_nodes, test_duration),
 (combined_event_dict, combined_num_nodes, combined_duration),
 nodes_not_in_train) = dataset_utils.load_enron_train_test()

for num_classes in range(1, 11):

    # Fitting the model to the train data
    train_node_membership, train_bp_mu, train_bp_alpha, train_bp_beta, train_block_pair_events = \
        model_utils.fit_community_model(train_event_dict, train_num_nodes, train_duration, num_classes)

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

    # Calculate per event log likelihood
    ll_per_event = model_utils.calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood,
                                                             test_event_dict, test_num_nodes)

    print(f"K: {num_classes} - ll: {ll_per_event:.3f}")
