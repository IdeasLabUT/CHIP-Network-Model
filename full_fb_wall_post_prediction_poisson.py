# -*- coding: utf-8 -*-
"""
"Exploratory Analysis of the Facebook Wall Posts Dataset using CHIP"

Here we fit CHIP to the largest connected component of the Facebook wall-post dataset in a train/test setting to
evaluate the log-likelihood of the model on the test set.

@author: Makan Arastuie
"""

import time
import pickle
import numpy as np
import dataset_utils
from scipy.stats import norm
import matplotlib.pyplot as plt
from plotting_utils import heatmap
import generative_model_utils as utils
import model_fitting_utils as fitting_utils
import parameter_estimation as estimate_utils
from spectral_clustering import spectral_cluster
import poisson_baseline_model_fitting as poisson_fitting


result_file_path = '/shared/Results/CommunityHawkes/pickles/full_fb_fit/poisson'

fit_chip = True
verbose = False
get_predictions = True
num_classes = 9

tic = time.time()
((train_event_dict, train_num_nodes, train_duration),
 (test_event_dict, test_num_nodes, test_duration),
 (combined_event_dict, combined_num_events, combined_duration),
 nodes_not_in_train) = dataset_utils.load_facebook_wall_2(largest_connected_component_only=True, train_percentage=0.8,
                                                          plot_growth=False, remove_nodes_not_in_train=True)

print(train_num_nodes + len(nodes_not_in_train))

toc = time.time()
print(f"Loaded the dataset in {toc - tic:.1f}s")

train_num_events = utils.num_events_in_event_dict(train_event_dict)
test_num_events = utils.num_events_in_event_dict(test_event_dict)

print("Train: ", "Num Nodes:", train_num_nodes, "Duration:", train_duration, "Num Edges:", train_num_events)
print("Test: ", "Num Nodes:", test_num_nodes, "Duration:", test_duration, "Num Edges:", test_num_events)
print("Total: ", "Num Nodes:", train_num_nodes + len(nodes_not_in_train),
      "Num Edges:", train_num_events + test_num_events)

# fit Facebook Wall-posts
if fit_chip:
    tic = time.time()
    train_agg_adj = utils.event_dict_to_aggregated_adjacency(train_num_nodes, train_event_dict)
    toc = time.time()

    if verbose:
        print(f"Generated aggregated adj in {toc - tic:.1f}s")

    tic_tot = time.time()
    tic = time.time()
    # Running spectral clustering
    train_node_membership = spectral_cluster(train_agg_adj, num_classes=num_classes, verbose=False,
                                             plot_eigenvalues=False, plot_save_path=result_file_path+'/plots')
    toc = time.time()

    print(f"Spectral clustering done in {toc - tic:.1f}s")

    if verbose:
        print("Community assignment prob:", np.unique(train_node_membership, return_counts=True)[1] / train_num_nodes)

    tic = time.time()
    # Fitting the model to the train data
    train_node_membership, train_bp_lambda, train_block_count_matrix = \
        poisson_fitting.fit_poisson_baseline_model(train_event_dict, train_num_nodes,
                                                   train_duration, num_classes, verbose=verbose)

    toc = time.time()

    print(f"Poisson param estimated in {toc - tic:.1f}s")

    if verbose:
        print("Lambda:")
        print(train_bp_lambda)

    # Save results
    with open(f'{result_file_path}/pred-param-k-{num_classes}.pckl', 'wb') as handle:
        pickle.dump([train_num_nodes, train_num_events, train_duration,
                     train_node_membership, train_block_count_matrix,
                     train_bp_lambda], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{result_file_path}/pred-param-k-{num_classes}.pckl', 'rb') as handle:
        [train_num_nodes, train_num_events, train_duration,
         train_node_membership, train_block_count_matrix,
         train_bp_lambda] = pickle.load(handle)


### calculate test log-likelihood
# Add nodes that were not in train to the largest block
combined_node_membership = fitting_utils.assign_node_membership_for_missing_nodes(train_node_membership,
                                                                                  nodes_not_in_train)

# Calculate log-likelihood given the entire dataset
combined_count_matrix = poisson_fitting.event_dict_to_block_pair_event_counts(combined_event_dict,
                                                                              combined_node_membership,
                                                                              num_classes)

combined_log_likelihood = poisson_fitting.calc_full_log_likelihood(combined_count_matrix, combined_node_membership,
                                                                   combined_duration, train_bp_lambda, num_classes)

# Calculate log-likelihood given the train dataset
train_log_likelihood = poisson_fitting.calc_full_log_likelihood(train_block_count_matrix, train_node_membership,
                                                                test_duration, train_bp_lambda, num_classes)

# Calculate per event log likelihood
ll_per_event = fitting_utils.calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood,
                                                           test_event_dict, test_num_nodes)

print("Test log_likelihood:", ll_per_event)



if get_predictions:
    test_block_pair_events = utils.event_dict_to_block_pair_events(test_event_dict,
                                                                   combined_node_membership,
                                                                   num_classes)
    test_block_pair_event_count = fitting_utils.compute_block_pair_total_event_count(test_block_pair_events,
                                                                                     num_classes)

    train_bp_size = utils.calc_block_pair_size(train_node_membership, num_classes)
    prediction_sample_mean = train_bp_lambda * test_duration * train_bp_size

    print("Sample mean and variance:")
    print(prediction_sample_mean)
    print("mean: ", np.mean(prediction_sample_mean))

    print("Event count:")
    print(test_block_pair_event_count)
    print(np.mean(test_block_pair_event_count))

    # Save results
    with open(f'{result_file_path}/predictions-k-{num_classes}.pckl', 'wb') as handle:
        pickle.dump([test_block_pair_events, test_block_pair_event_count,
                     prediction_sample_mean], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{result_file_path}/predictions-k-{num_classes}.pckl', 'rb') as handle:
        [test_block_pair_events, test_block_pair_event_count,
         prediction_sample_mean] = pickle.load(handle)


percentage_within = []
for i in np.arange(0, 1.01, 0.01):
    # lower_ci, upper_ci = norm.interval(i, loc=prediction_sample_mean, scale=np.sqrt(prediction_sample_mean))

    lower_ci, upper_ci = norm.interval(1 - (1 - i) / (num_classes ** 2), loc=prediction_sample_mean,
                                       scale=np.sqrt(prediction_sample_mean))
    block_pairs_within_interval = np.logical_and(test_block_pair_event_count >= lower_ci,
                                                 test_block_pair_event_count <= upper_ci)

    percentage_within.append(np.sum(block_pairs_within_interval) / (num_classes ** 2))

plt.plot(np.arange(0, 1.01, 0.01), percentage_within)
plt.ylabel("Percentage of Block-pair Event Count Within CI")
plt.xlabel("Width of Confidence Interval (CI)")
plt.ylim((0, 1))
plt.savefig(f'{result_file_path}/plots/prediction-interval.pdf', format='pdf')
plt.show()
