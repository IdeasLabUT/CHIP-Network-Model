import numpy as np
import dataset_utils
import generative_model_utils as utils
import parameter_estimation as estimate_utils
from spectral_clustering import spectral_cluster


def fit_community_model(event_dict, num_nodes, duration, num_classes):
    agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)
    # adj = utils.event_dict_to_adjacency(num_nodes, event_dict)

    # Running spectral clustering
    node_membership = spectral_cluster(agg_adj, num_classes)

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

    return node_membership, bp_mu, bp_alpha, bp_beta, block_pair_events


def calc_full_log_likelihood(block_pair_events, node_membership, bp_mu, bp_alpha, bp_beta, duration, num_classes):
    log_likelihood = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            log_likelihood += estimate_utils.full_log_likelihood(block_pair_events[b_i][b_j],
                                                                       bp_mu[b_i, b_j], bp_alpha[b_i, b_j],
                                                                       bp_beta[b_i, b_j], duration,
                                                                       block_pair_size=bp_size)

    return log_likelihood


def assign_node_membership_for_missing_nodes(node_membership, missing_nodes):
    class_idx, class_count = np.unique(node_membership, return_counts=True)
    largest_class_idx = class_idx[np.argmax(class_count)]

    combined_node_membership = np.copy(node_membership)

    missing_nodes.sort()
    for n in missing_nodes:
        combined_node_membership = np.insert(combined_node_membership, n, largest_class_idx)

    return combined_node_membership


def calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood, test_event_dict, test_num_nodes):
    test_num_events = np.sum(utils.event_dict_to_aggregated_adjacency(test_num_nodes, test_event_dict))
    return (combined_log_likelihood - train_log_likelihood) / test_num_events

