# -*- coding: utf-8 -*-
"""
"Community Detection with Varying T, n and k"

** simulation AS3: fix parameter values, draw a heatmap in each case. The following setup is such that the lower left
corner will be the hard regime, so highest mis-clustering rate and and right top corner will be the easy regime with
lower mis-clustering rate.

(a) fix $n$ increasing $T$ and decreasing $k$.
(b) fix $T$, increasing $n$ and decreasing $k$.
(c) fix $k$, increasing $n$ and increasing $T$.

@author: Makan Arastuie
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import heatmap
from joblib import Parallel, delayed
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster


def test_spectral_clustering_on_generative_model(n, t, k):
    params = {'number_of_nodes': n,
              'end_time': t,
              'class_probabilities': np.ones(k) / k,
              'alpha': 0.06,
              'beta': 0.08,
              'mu_diag': 0.085,
              'mu_off_diag': 0.065,
              'scale': False,
              'n_cores': 1}

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params)

    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(len(true_class_assignments), event_dict)
    agg_adj_pred = spectral_cluster(agg_adj, num_classes=k)
    agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    return agg_adj_sc_rand


result_file_path = '/shared/Results/CommunityHawkes/pickles/AS3'

plot_only = True

# Number of test values for all variable must be the same
n_range = [2048, 1024, 512, 256, 128, 64]
t_range = [1024, 512, 256, 128, 64, 32]
k_range = [12, 10, 8, 6, 4, 2]

num_test_values = len(n_range)

fixed_n = 256
fixed_t = 64
fixed_k = 8

num_simulation_per_duration = 30
n_cores = 30

for fixed_var in ['n', 't', 'k']:
    print("Fixing:", fixed_var)

    n_range_to_test = n_range
    t_range_to_test = t_range
    k_range_to_test = k_range

    if fixed_var == 'n':
        n_range_to_test = [fixed_n]

        fixed_value = fixed_n
        ylables = t_range
        xlables = k_range
        xlab = "K"
        ylab = "T"

    elif fixed_var == 't':
        t_range_to_test = [fixed_t]

        fixed_value = fixed_t
        ylables = n_range
        xlables = k_range
        xlab = "K"
        ylab = "N"

    else:
        k_range_to_test = [fixed_k]
        t_range_to_test = t_range_to_test[::-1]

        fixed_value = fixed_k
        ylables = n_range
        xlables = t_range[::-1]
        xlab = "T"
        ylab = "N"

    if not plot_only:
        mean_sc_rand_scores = []
        mean_sc_rand_scores_err = []

        cnt = 0
        for n in n_range_to_test:
            for t in t_range_to_test:
                for k in k_range_to_test:
                    results = Parallel(n_jobs=n_cores)(delayed(test_spectral_clustering_on_generative_model)
                                                       (n, t, k) for i in range(num_simulation_per_duration))

                    cnt += 1
                    print(f"Done simulating {cnt} of {num_test_values ** 2}.")

                    results = np.asarray(results, dtype=np.float)

                    mean_sc_rand_scores.append(np.mean(results))
                    mean_sc_rand_scores_err.append(2 * np.std(results) / np.sqrt(len(results)))

        mean_sc_rand_scores = np.reshape(mean_sc_rand_scores, (num_test_values, num_test_values))
        mean_sc_rand_scores_err = np.reshape(mean_sc_rand_scores_err, (num_test_values, num_test_values))

        # Save results
        with open(f'{result_file_path}/all_sims-fixed-{fixed_var}.pckl', 'wb') as handle:
            pickle.dump([mean_sc_rand_scores, mean_sc_rand_scores_err], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{result_file_path}/all_sims-fixed-{fixed_var}.pckl', 'rb') as handle:
        [mean_sc_rand_scores, mean_sc_rand_scores_err] = pickle.load(handle)

    print(f"community model fixed {fixed_var}: {fixed_value}")
    print(f"rand:", mean_sc_rand_scores)
    print(f"rand error:", mean_sc_rand_scores_err)

    # Plot Results
    fig, ax = plt.subplots()

    im, _ = heatmap(mean_sc_rand_scores, ylables, xlables, ax=ax, cmap="coolwarm",
                    cbarlabel=f"Adjusted Rand Score")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(n_range)):
    #     for j in range(len(n_range)):
    #         text = ax.text(j, i, np.format_float_scientific(mean_sc_rand_scores_err[i, j], exp_digits=1, precision=1),
    #                        ha="center", va="center", color="w")

    plt.ylabel(ylab, fontsize=16)
    plt.xlabel(xlab, fontsize=16)
    # ax.set_title(f"CHIP SC AS3 Fixed {fixed_var.upper()}: {fixed_value}")
    fig.tight_layout()
    plt.savefig(f"{result_file_path}/plots/as3-fixed-{fixed_var}.pdf")
    # plt.show()

