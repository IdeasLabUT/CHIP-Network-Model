# -*- coding: utf-8 -*-
"""
Empirically analyzing the efficacy of local search on rand score after spectral clustering, to see at which point local
search is no longer necessary.

@author: Makan Arastuie
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster
from chip_local_search import chip_local_search


result_file_path = '/shared/Results/CommunityHawkes/pickles/local_search_efficacy'
pickle_file_name = 'ls_efficacy'
plot_name = 'ls_efficacy'
plot_only = True

num_simulation_per_duration = 15
sim_n_cores = 1
per_sim_n_cores = 35

# number_of_nodes_list = [32, 64, 128, 256, 512]
number_of_nodes_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
n_classes = 4
duration = 100

params = {'alpha': 0.16,
          'beta': 0.18,
          'mu_diag': 0.175,
          'mu_off_diag': 0.12,
          'scale': False,
          'end_time': duration,
          'class_probabilities': np.ones(n_classes) / n_classes,
          'n_cores': per_sim_n_cores}


def test_spectral_clustering_on_generative_model(n_nodes):
    params['number_of_nodes'] = n_nodes

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params)

    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
    sc_node_membership = spectral_cluster(agg_adj, num_classes=n_classes, verbose=False)
    sc_rand = adjusted_rand_score(true_class_assignments, sc_node_membership)

    ls_node_membership = chip_local_search(event_dict, n_classes, sc_node_membership, duration,
                                           max_iter=10000, n_cores=per_sim_n_cores, verbose=False)
    ls_rand = adjusted_rand_score(true_class_assignments, ls_node_membership)

    return sc_rand, ls_rand


# Save results

if not plot_only:
    mean_sc_rand = []
    mean_sc_rand_err = []

    mean_ls_rand = []
    mean_ls_rand_err = []

    for number_of_nodes in number_of_nodes_list:

        if sim_n_cores > 1:
            results = Parallel(n_jobs=sim_n_cores)(delayed(test_spectral_clustering_on_generative_model)
                                                   (number_of_nodes)
                                                   for i in range(num_simulation_per_duration))
        else:
            results = []

            for i in range(num_simulation_per_duration):
                results.append(test_spectral_clustering_on_generative_model(number_of_nodes))

        # each row is adj_sc_rand, agg_adj_sc_rand
        results = np.asarray(results, dtype=np.float)

        mean_sc_rand.append(np.mean(results[:, 0]))
        mean_sc_rand_err.append(2 * np.std(results[:, 0]) / np.sqrt(len(results[:, 0])))

        mean_ls_rand.append(np.mean(results[:, 1]))
        mean_ls_rand_err.append(2 * np.std(results[:, 1]) / np.sqrt(len(results[:, 1])))

        print(f"Done simulating with {number_of_nodes} nodes.")
        print(f"SC Rand: {mean_sc_rand[-1]}")
        print(f"SC LS Rand: {mean_ls_rand[-1]}")

    with open(f'{result_file_path}/{pickle_file_name}', 'wb') as handle:
        pickle.dump([mean_sc_rand,
                     mean_sc_rand_err,
                     mean_ls_rand,
                     mean_ls_rand_err,
                     params], handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{result_file_path}/{pickle_file_name}', 'rb') as handle:
    [mean_sc_rand,
     mean_sc_rand_err,
     mean_ls_rand,
     mean_ls_rand_err,
     params] = pickle.load(handle)


print(f"community model:")
print("Number of nodes:", number_of_nodes_list)
print(f"SC Rand:", mean_sc_rand)
print(f"LS Rand:", mean_ls_rand)
print(params)

# Plot Results
fig, ax = plt.subplots()
ind = np.arange(len(number_of_nodes_list))    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, mean_sc_rand, width, color='r', yerr=mean_sc_rand_err)
p2 = ax.bar(ind + width, mean_ls_rand, width, color='b', yerr=mean_ls_rand_err)

# ax.set_title(f'Community Model\'s Mean Adjusted Rand Scores')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(number_of_nodes_list, fontsize=12)
ax.set_ylim(0, 1)
ax.tick_params(labelsize=12)

ax.legend((p1[0], p2[0]), (f"SC", f"SC + LS"), fontsize=14)
plt.ylabel("Mean Adjusted Rand Score", fontsize=16)
plt.xlabel("Number of Nodes", fontsize=16)

ax.autoscale_view()

plt.savefig(f"{result_file_path}/plots/{plot_name}.pdf")
plt.show()
