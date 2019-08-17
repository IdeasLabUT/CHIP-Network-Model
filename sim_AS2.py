# -*- coding: utf-8 -*-
"""
"Effects of Diagonal and Off-diagonal Mu's on Community Detection"

** simulation AS2: fix $n,k,T$, then:

(a) Increase mu_diag and mu_off_diag such that the ratio mu_off_diag/mu_diag remains the same.
(b) Hold mu_off_diag fixed and only increase mu_diag slowly.

Expectation: We should see accuracy increase in both these cases. When mu_diag/mu_off_diag ratio is low, the algorithms
will do poorly, but as the ratio increases there is more signal and the algorithm will do well and go all the way to 1.

@author: Makan Arastuie
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from dataset_utils import get_script_path
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster


result_file_path = f'{get_script_path()}/storage/results/AS2'

also_use_unweighted_adjacency = True

sim_type = 'a'
# sim_type = 'b'

plot_only = True

plot_name = "fixed_ratio" if sim_type == 'a' else "increase_mu_diag"

# a_scalars_to_test = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
a_scalars_to_test = [1, 5, 10, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
b_scalars_to_test = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]

n_classes = 4
num_simulation_per_duration = 100
n_cores = 6

scalars_to_test = a_scalars_to_test if sim_type == 'a' else b_scalars_to_test


def test_spectral_clustering_on_generative_model(scalar):
    params = {'alpha': 0.05,
              'beta': 0.08,
              'mu_diag': 0.00075 * scalar,
              'mu_off_diag': 0.00035 if sim_type == 'b' else 0.00035 * scalar,
              'scale': False,
              'number_of_nodes': 256}

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params)
    num_nodes = len(true_class_assignments)
    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)
    agg_adj_pred = spectral_cluster(agg_adj, num_classes=n_classes)
    agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    if not also_use_unweighted_adjacency:
        return agg_adj_sc_rand

    # Spectral clustering on aggregated adjacency matrix
    adj = utils.event_dict_to_adjacency(num_nodes, event_dict)
    adj_pred = spectral_cluster(adj, num_classes=n_classes)
    adj_sc_rand = adjusted_rand_score(true_class_assignments, adj_pred)

    return agg_adj_sc_rand, adj_sc_rand, np.sum(adj)/(num_nodes ** 2)


if not plot_only:
    agg_adj_mean_sc_rand_scores = []
    agg_adj_mean_sc_rand_scores_err = []

    adj_mean_sc_rand_scores = []
    adj_mean_sc_rand_scores_err = []

    mean_proportion_ones_in_adj = []
    mean_proportion_ones_in_adj_err = []

    for scalar in scalars_to_test:

        results = Parallel(n_jobs=n_cores)(delayed(test_spectral_clustering_on_generative_model)
                                           (scalar) for i in range(num_simulation_per_duration))

        print(f"Done simulating with {scalar} scalar.")

        results = np.asarray(results, dtype=np.float)

        if also_use_unweighted_adjacency:
            agg_adj_mean_sc_rand_scores.append(np.mean(results[:, 0]))
            agg_adj_mean_sc_rand_scores_err.append(2 * np.std(results[:, 0]) / np.sqrt(len(results[:, 0])))

            adj_mean_sc_rand_scores.append(np.mean(results[:, 1]))
            adj_mean_sc_rand_scores_err.append(2 * np.std(results[:, 1]) / np.sqrt(len(results[:, 1])))

            mean_proportion_ones_in_adj.append(np.mean(results[:, 2]))
            mean_proportion_ones_in_adj_err.append(2 * np.std(results[:, 2]) / np.sqrt(len(results[:, 2])))

            # Save results
            with open(f'{result_file_path}/all_sims-{sim_type}-w-adj.pckl', 'wb') as handle:
                pickle.dump([agg_adj_mean_sc_rand_scores, agg_adj_mean_sc_rand_scores_err,
                             adj_mean_sc_rand_scores, adj_mean_sc_rand_scores_err,
                             mean_proportion_ones_in_adj,
                             mean_proportion_ones_in_adj_err], handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            agg_adj_mean_sc_rand_scores.append(np.mean(results))
            agg_adj_mean_sc_rand_scores_err.append(2 * np.std(results) / np.sqrt(len(results)))

            # Save results
            with open(f'{result_file_path}/all_sims-{sim_type}.pckl', 'wb') as handle:
                pickle.dump([agg_adj_mean_sc_rand_scores,
                             agg_adj_mean_sc_rand_scores_err], handle, protocol=pickle.HIGHEST_PROTOCOL)


if also_use_unweighted_adjacency:
    with open(f'{result_file_path}/all_sims-{sim_type}-w-adj.pckl', 'rb') as handle:
        [agg_adj_mean_sc_rand_scores, agg_adj_mean_sc_rand_scores_err,
         adj_mean_sc_rand_scores, adj_mean_sc_rand_scores_err,
         mean_proportion_ones_in_adj, mean_proportion_ones_in_adj_err] = pickle.load(handle)
else:
    with open(f'{result_file_path}/all_sims-{sim_type}.pckl', 'rb') as handle:
        [agg_adj_mean_sc_rand_scores, agg_adj_mean_sc_rand_scores_err] = pickle.load(handle)


print(f"community model:")
print("Number of nodes:", scalars_to_test)

print(f"Agg Adj rand:", agg_adj_mean_sc_rand_scores)
print(f"Agg Adj rand error:", agg_adj_mean_sc_rand_scores_err)

if also_use_unweighted_adjacency:
    print(f"Adj rand:", adj_mean_sc_rand_scores)
    print(f"Adj rand error:", adj_mean_sc_rand_scores_err)

    print(f"Ones proportion:", mean_proportion_ones_in_adj)
    print(f"Ones proportion error:", mean_proportion_ones_in_adj_err)


# Plot Results
if not also_use_unweighted_adjacency:
    fig, ax = plt.subplots()
    ind = np.arange(len(scalars_to_test))    # the x locations for the groups
    p1 = ax.bar(ind, agg_adj_mean_sc_rand_scores, color='c', yerr=agg_adj_mean_sc_rand_scores_err)

    # ax.set_title(f'AS2 {sim_type} Community Model\'s Mean Adjusted Rand Scores')
    ax.set_xticks(ind)
    ax.tick_params(labelsize=12)
    ax.set_xticklabels(scalars_to_test, fontsize=12)
    plt.xlabel("Scalars", fontsize=16)
    plt.ylabel("Mean Adjusted Rand Score", fontsize=16)

    ax.set_ylim(0, 1)

    ax.autoscale_view()

    plt.savefig(result_file_path + "/plots/" + plot_name + ".pdf")
    plt.show()
else:
    w, h = plt.figaspect(.3)
    # fig = plt.Figure()
    fig, ax = plt.subplots(figsize=(w, h))
    ind = np.arange(len(agg_adj_mean_sc_rand_scores))    # the x locations for the groups
    width = 0.35         # the width of the bars
    p1 = ax.bar(ind, agg_adj_mean_sc_rand_scores, width, color='b', yerr=agg_adj_mean_sc_rand_scores_err)
    p2 = ax.bar(ind + width, adj_mean_sc_rand_scores, width, color='r', yerr=adj_mean_sc_rand_scores_err)
    plt.axhline(y=1, color='black', linestyle='-')

    # ax.set_title(f'Community Model\'s Mean Adjusted Rand Scores')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(a_scalars_to_test, fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.2], [0.2])
    ax.tick_params(labelsize=12)

    rects = ax.patches
    for rect, label in zip(rects, mean_proportion_ones_in_adj):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width(), 1.03, f"{label:.3f}",
                ha='center', va='bottom', rotation='vertical', fontsize=12)

    ax.legend((p1[0], p2[0]), ("Weighted Adjacency", "Unweighted Adjacency"), fontsize=14,
              bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=4)
    plt.ylabel("Mean Adjusted Rand Score", fontsize=16)
    plt.xlabel("Scalars", fontsize=16)

    ax.autoscale_view()

    plt.savefig(f"{result_file_path}/plots/agg-vs-adj-density.pdf")
    plt.show()