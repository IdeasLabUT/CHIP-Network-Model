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


result_file_path = '/shared/Results/CommunityHawkes/pickles/full_fb_fit'

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

# ((train_event_dict, train_num_nodes, train_duration),
#  (test_event_dict, test_num_nodes, test_duration),
#  (combined_event_dict, combined_num_events, combined_duration),
#  nodes_not_in_train) = dataset_utils.load_facebook_wall(largest_connected_component_only=True, train_percentage=0.8)

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
    bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio = fitting_utils.estimate_bp_hawkes_params(train_event_dict,
                                                                                            train_node_membership,
                                                                                            train_duration,
                                                                                            num_classes)
    toc = time.time()

    print(f"Hawkes params estimated in {toc - tic:.1f}s")

    if verbose:
        print("Mu:")
        print(bp_mu)

        print("Ratio:")
        print(bp_alpha_beta_ratio)

        print("Alpha:")
        print(bp_alpha)

        print("Beta:")
        print(bp_beta)

    train_hawkes_params = {'mu': bp_mu,
                           'alpha_beta_ratio': bp_alpha_beta_ratio,
                           'alpha': bp_alpha,
                           'beta': bp_beta}

    # Save results
    with open(f'{result_file_path}/pred-all-model-params-k-{num_classes}.pckl', 'wb') as handle:
        pickle.dump([train_num_nodes, train_num_events, train_duration,
                     train_node_membership,
                     train_hawkes_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{result_file_path}/pred-all-model-params-k-{num_classes}.pckl', 'rb') as handle:
        [train_num_nodes, train_num_events, train_duration,
         train_node_membership,
         train_hawkes_params] = pickle.load(handle)


### calculate test log-likelihood
combined_node_membership = fitting_utils.assign_node_membership_for_missing_nodes(train_node_membership,
                                                                                  nodes_not_in_train)

# # Calculate log-likelihood given the entire dataset
# combined_block_pair_events = utils.event_dict_to_block_pair_events(combined_event_dict,
#                                                                    combined_node_membership,
#                                                                    num_classes)
#
# combined_log_likelihood = fitting_utils.calc_full_log_likelihood(combined_block_pair_events,
#                                                                  combined_node_membership,
#                                                                  train_hawkes_params['mu'],
#                                                                  train_hawkes_params['alpha'],
#                                                                  train_hawkes_params['beta'],
#                                                                  combined_duration, num_classes)
#
# # Calculate log-likelihood given the train dataset
# train_block_pair_events = utils.event_dict_to_block_pair_events(train_event_dict, train_node_membership, num_classes)
# train_log_likelihood = fitting_utils.calc_full_log_likelihood(train_block_pair_events, train_node_membership,
#                                                               train_hawkes_params['mu'],
#                                                               train_hawkes_params['alpha'],
#                                                               train_hawkes_params['beta'],
#                                                               train_duration, num_classes)
#
# # Calculate per event log likelihood
# ll_per_event = fitting_utils.calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood,
#                                                            test_event_dict, test_num_nodes)
#
# print("Test log_likelihood:", ll_per_event)


if get_predictions:
    # Calculate mean and variance for block-pair events counts in test dataset
    test_block_pair_events = utils.event_dict_to_block_pair_events(test_event_dict,
                                                                   combined_node_membership,
                                                                   num_classes)

    test_event_count_mean, test_event_count_variance = \
        fitting_utils.compute_block_pair_event_count_empirical_mean_and_variance(test_block_pair_events,
                                                                                 combined_node_membership,
                                                                                 num_classes)

    prediction_sample_mean, prediction_sample_var, test_block_pair_event_count = \
        fitting_utils.compute_prediction_mean_and_variance_for_block_pair_event_count(train_hawkes_params['mu'],
                                                                                      train_hawkes_params['alpha_beta_ratio'],
                                                                                      test_block_pair_events,
                                                                                      train_node_membership,
                                                                                      num_classes, train_duration,
                                                                                      test_duration)

    print("Sample mean:")
    print(prediction_sample_mean)
    print(np.mean(prediction_sample_mean))

    print("Sample var:")
    print(prediction_sample_var)

    print("Event count:")
    print(test_block_pair_event_count)
    print(np.mean(test_block_pair_event_count))

    # Save results
    with open(f'{result_file_path}/predictions-k-{num_classes}.pckl', 'wb') as handle:
        pickle.dump([test_block_pair_events, test_event_count_mean, test_event_count_variance,
                     prediction_sample_mean, prediction_sample_var,
                     test_block_pair_event_count], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{result_file_path}/predictions-k-{num_classes}.pckl', 'rb') as handle:
        [test_block_pair_events, test_event_count_mean, test_event_count_variance,
         prediction_sample_mean, prediction_sample_var,
         test_block_pair_event_count] = pickle.load(handle)


# print("Sample mean:")
# print(prediction_sample_mean)
# print(np.mean(prediction_sample_mean))
#
# print("Sample var:")
# print(prediction_sample_var)
#
# print("Event count:")
# print(test_block_pair_event_count)
# print(np.mean(test_block_pair_event_count))

percentage_within = []
for i in np.arange(0, 1.01, 0.01):
    lower_ci, upper_ci = norm.interval(i, loc=prediction_sample_mean, scale=np.sqrt(prediction_sample_var))

    # lower_ci, upper_ci = norm.interval(1 - (1 - i) / (num_classes ** 2), loc=prediction_sample_mean,
    #                                    scale=np.sqrt(prediction_sample_var))
    block_pairs_within_interval = np.logical_and(test_block_pair_event_count >= lower_ci,
                                                 test_block_pair_event_count <= upper_ci)

    # print(i, block_pairs_within_interval)

    percentage_within.append(np.sum(block_pairs_within_interval) / (num_classes ** 2))

# print(test_block_pair_event_count)
plt.plot(np.arange(0, 1.01, 0.01), percentage_within)
plt.ylabel("Percentage of Block-pair Event Count Within CI")
plt.xlabel("Width of Confidence Interval (CI)")
plt.ylim((0, 1))
plt.show()
plt.savefig(f'{result_file_path}/plots/prediction-interval.pdf', format='pdf')
