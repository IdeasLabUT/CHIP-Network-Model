# -*- coding: utf-8 -*-
"""
"Exploratory Analysis of the Facebook Wall Posts Dataset using CHIP", comparison of the log-likelihoods of
the weighted vs. unweighted adjacency matrices.

Here we fit CHIP to the largest connected component of the Facebook wall-post dataset in a train/test setting to
evaluate the log-likelihood of the model on the test set.

@author: Makan Arastuie
"""

import time
import pickle
import numpy as np
import dataset_utils
import matplotlib.pyplot as plt
from plotting_utils import heatmap
import generative_model_utils as utils
import model_fitting_utils as fitting_utils
import parameter_estimation as estimate_utils
from spectral_clustering import spectral_cluster


result_file_path = '/shared/Results/CommunityHawkes/pickles/fb_chip_fit'

use_agg_adj = True
fit_chip = True
load_fb = True
plot_hawkes_params = False
plot_node_membership = False
simulate_chip = True
verbose = False
num_classes = 10

file_names_append = "agg_adj" if use_agg_adj else "adj"

tic = time.time()
((train_event_dict, train_num_nodes, train_duration),
 (test_event_dict, test_num_nodes, test_duration),
 (combined_event_dict, combined_num_events, combined_duration),
 nodes_not_in_train) = dataset_utils.load_facebook_wall(largest_connected_component_only=True, train_percentage=0.8)
toc = time.time()
print(f"Loaded the dataset in {toc - tic:.1f}s")

train_num_events = utils.num_events_in_event_dict(train_event_dict)
test_num_events = utils.num_events_in_event_dict(test_event_dict)
# if verbose:
print("Train: ", "Num Nodes:", train_num_nodes, "Duration:", train_duration, "Num Edges:", train_num_events)
print("Test: ", "Num Nodes:", test_num_nodes, "Duration:", test_duration, "Num Edges:", test_num_events)

# fit Facebook Wall-posts
if fit_chip:
    tic = time.time()
    train_agg_adj = utils.event_dict_to_aggregated_adjacency(train_num_nodes, train_event_dict)

    if not use_agg_adj:
        train_adj = utils.event_dict_to_adjacency(train_num_nodes, train_event_dict)
    toc = time.time()

    if verbose:
        print(f"Generated aggregated adj in {toc - tic:.1f}s")

    tic_tot = time.time()
    tic = time.time()
    # Running spectral clustering
    if use_agg_adj:
        train_node_membership = spectral_cluster(train_agg_adj, num_classes=num_classes, verbose=False,
                                                 plot_eigenvalues=False)
    else:
        train_node_membership = spectral_cluster(train_adj, num_classes=num_classes, verbose=False,
                                                 plot_eigenvalues=False)
    toc = time.time()

    print(f"Spectral clustering done in {toc - tic:.1f}s")

    if verbose:
        print("Community assignment prob:", np.unique(train_node_membership, return_counts=True)[1] / train_num_nodes)

    tic = time.time()
    bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(train_agg_adj, train_node_membership,
                                                                            train_duration,
                                                                            1e-10 / train_duration)
    toc = time.time()

    print(f"Mu and m estimated in {toc - tic:.1f}s")

    if verbose:
        print("Mu:")
        print(bp_mu)
        print("Ratio:")
        print(bp_alpha_beta_ratio)

    print("\nStart Beta estimation:")

    tic = time.time()
    bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)
    train_block_pair_events = utils.event_dict_to_block_pair_events(train_event_dict, train_node_membership, num_classes)
    bp_size = utils.calc_block_pair_size(train_node_membership, num_classes)

    cnt = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(train_block_pair_events[b_i][b_j],
                                                                            bp_mu[b_i, b_j],
                                                                            bp_alpha_beta_ratio[b_i, b_j],
                                                                            train_duration, bp_size[b_i, b_j])
            cnt += 1
            print(f"{100 * cnt / num_classes ** 2:0.2f}% Done.", end='\r')

    bp_alpha = bp_alpha_beta_ratio * bp_beta
    toc = time.time()
    toc_tot = time.time()

    print(f"Beta estimated in {toc - tic:.1f}s")
    print(f"Total computation time: {toc_tot - tic_tot:.1f}s")

    if verbose:
        print("Alpha")
        print(bp_alpha)

        print("Beta")
        print(bp_beta)

    hawkes_params = {'mu': bp_mu,
                     'alpha': bp_alpha,
                     'beta': bp_beta,
                     'alpha_beta_ratio': bp_alpha_beta_ratio}




    # Save results
    with open(f'{result_file_path}/pred-all-model-params-k-{num_classes}-{file_names_append}.pckl', 'wb') as handle:
        pickle.dump([train_num_nodes, train_num_events, train_duration,
                     train_node_membership,
                     hawkes_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{result_file_path}/pred-all-model-params-k-{num_classes}-{file_names_append}.pckl', 'rb') as handle:
        [train_num_nodes, train_num_events, train_duration,
         train_node_membership,
         hawkes_params] = pickle.load(handle)

bp_mu, bp_alpha, bp_beta = hawkes_params['mu'], hawkes_params['alpha'], hawkes_params['beta']



### calculate test log-likelihood
combined_node_membership = fitting_utils.assign_node_membership_for_missing_nodes(train_node_membership,
                                                                                      nodes_not_in_train)

# Calculate log-likelihood given the entire dataset
combined_block_pair_events = utils.event_dict_to_block_pair_events(combined_event_dict,
                                                                   combined_node_membership,
                                                                   num_classes)

combined_log_likelihood = fitting_utils.calc_full_log_likelihood(combined_block_pair_events,
                                                                 combined_node_membership,
                                                                 bp_mu, bp_alpha, bp_beta,
                                                                 combined_duration, num_classes)

# Calculate log-likelihood given the train dataset
train_block_pair_events = utils.event_dict_to_block_pair_events(train_event_dict, train_node_membership, num_classes)
train_log_likelihood = fitting_utils.calc_full_log_likelihood(train_block_pair_events, train_node_membership,
                                                              bp_mu, bp_alpha, bp_beta,
                                                              train_duration, num_classes)

# Calculate per event log likelihood
ll_per_event = fitting_utils.calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood,
                                                           test_event_dict, test_num_nodes)

print("Test log_likelihood:", ll_per_event)




# Add nodes that were not in train to the largest block
combined_node_membership = fitting_utils.assign_node_membership_for_missing_nodes(train_node_membership,
                                                                                  nodes_not_in_train)

# test_block_pair_events = utils.event_dict_to_block_pair_events(test_event_dict, combined_node_membership, num_classes)
test_block_pair_events = utils.event_dict_to_block_pair_events(test_event_dict, combined_node_membership, num_classes)

bp_size = utils.calc_block_pair_size(combined_node_membership, num_classes)
test_mean_num_events_bp = np.zeros((num_classes, num_classes))
for i in range(num_classes):
    for j in range(num_classes):
        test_mean_num_events_bp[i, j] = len(test_block_pair_events[i][j]) / bp_size[i, j]

# print(test_mean_num_events_bp)


pred_mean_num_events_bp = (hawkes_params['mu'] * test_duration) / (1 - (hawkes_params['alpha'] / hawkes_params['beta']))

pred_se = np.abs(pred_mean_num_events_bp - test_mean_num_events_bp)

fig, ax = plt.subplots()
labels = np.arange(1, num_classes + 1)
im, _ = heatmap(pred_se, labels, labels, ax=ax, cmap="Greys",
                cbarlabel="Number of Events Per Node Pair")

fig.tight_layout()
# plt.show()
