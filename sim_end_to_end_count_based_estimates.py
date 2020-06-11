# -*- coding: utf-8 -*-
"""
"CHIP end-to-end parameter Estimation"

Empirically analyzing the consistency of the CHIP parameter estimators.

@author: Makan Arastuie
"""

import os
import copy
import pickle
import numpy as np
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
import model_fitting_utils as model_utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster
from parameter_estimation import estimate_hawkes_from_counts
from sklearn.linear_model import LinearRegression

result_file_path = join(os.sep, '/shared', 'Results', 'CommunityHawkes', 'pickles',
                        'end_to_end_count_based_estimate', 'up_to_256_scaled')
Path(join(result_file_path, 'plots')).mkdir(parents=True, exist_ok=True)

run_analysis = False
run_plotting = False
run_regression = True

# # sim params
# end_time = 100
# mu_diag = 1.85
# mu_off_diag = 1.65
#
# alpha_diag = 0.75
# alpha_off_diag = 0.65
#
# beta_diag = 0.85
# beta_off_diag = 0.95


# # sim params
# end_time = 100
# mu_diag = 0.07
# mu_off_diag = 0.06
#
# alpha_diag = 0.06
# alpha_off_diag = 0.05
#
# beta_diag = 0.07
# beta_off_diag = 0.08


# # # sim params
# end_time = 100
# mu_diag = 0.85
# mu_off_diag = 0.83
#
# alpha_diag = 0.75
# alpha_off_diag = 0.73
#
# beta_diag = 0.85
# beta_off_diag = 0.87

# # # sim params
# end_time = 100
# mu_diag = 0.85
# mu_off_diag = 0.83
#
# alpha_diag = 0.74
# alpha_off_diag = 0.73
#
# beta_diag = 0.86
# beta_off_diag = 0.87

# # # sim params
# end_time = 100
# mu_diag = 0.085
# mu_off_diag = 0.075
#
# alpha_diag = 0.075
# alpha_off_diag = 0.065
#
# beta_diag = 0.085
# beta_off_diag = 0.095

asy_scalar = 100

# # sim params
end_time = 100 * asy_scalar
mu_diag = 0.11 / asy_scalar
mu_off_diag = 0.1 / asy_scalar

alpha_diag = 0.11
alpha_off_diag = 0.09

beta_diag = 0.14
beta_off_diag = 0.16


class_probs = [0.25, 0.25, 0.25, 0.25]
# num_nodes_to_test = [8, 16, 32, 64, 128, 256]
# num_nodes_to_test = [16, 32, 64, 128, 256, 512]
# num_nodes_to_test = np.logspace(4, 7, num=7, dtype=np.int32, base=2)  # 128 - 7
# num_nodes_to_test = np.logspace(5, 7, num=11, dtype=np.int32, base=2)  # 128 - 11
num_nodes_to_test = np.logspace(5, 8, num=7, dtype=np.int32, base=2)  # 256
# num_nodes_to_test = np.logspace(4, 6, num=5, dtype=np.int32, base=2)  # test
num_simulations = 100
n_cores = 6
n_classes = len(class_probs)


def calc_mean_and_error_of_count_estiamte(n_nodes):
    params = {'number_of_nodes': n_nodes,
              'class_probabilities': class_probs,
              'end_time': end_time,
              'mu_diag': mu_diag,
              'mu_off_diag': mu_off_diag,
              'alpha': alpha_off_diag,
              'alpha_diag': alpha_diag,
              'beta': beta_off_diag,
              'beta_diag': beta_diag,
              'scale': False}

    event_dict, true_node_membership = utils.simulate_community_hawkes(params)

    invalid_cluster = True

    while invalid_cluster:
        # Spectral clustering on aggregated adjacency matrix
        agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
        node_membership = spectral_cluster(agg_adj, num_classes=n_classes, verbose=False)
        unique_vals, cnts = np.unique(node_membership, return_counts=True)
        invalid_cluster = len(unique_vals) != n_classes
        if len(unique_vals) != n_classes:
            print("a;ldskfjasdlf")
            print(unique_vals, cnts)

    sc_rand = adjusted_rand_score(true_node_membership, node_membership)
    sc_rand = np.zeros((n_classes, n_classes)) + sc_rand  # match the shape of other params to retrieve easily

    # param estimation with estimated communities
    bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio = model_utils.estimate_bp_hawkes_params(event_dict,
                                                                                          node_membership,
                                                                                          end_time,
                                                                                          n_classes)
    # param estimation with known communities. k_ is for known_
    k_bp_mu, k_bp_alpha, k_bp_beta, k_bp_alpha_beta_ratio = model_utils.estimate_bp_hawkes_params(event_dict,
                                                                                                  true_node_membership,
                                                                                                  end_time,
                                                                                                  n_classes)
    return bp_mu, bp_alpha_beta_ratio, bp_alpha, bp_beta, sc_rand, k_bp_mu, k_bp_alpha_beta_ratio, k_bp_alpha, k_bp_beta


true_mu = np.zeros((n_classes, n_classes)) + mu_off_diag
np.fill_diagonal(true_mu, mu_diag)

true_alpha = np.zeros((n_classes, n_classes)) + alpha_off_diag
np.fill_diagonal(true_alpha, alpha_diag)

true_beta = np.zeros((n_classes, n_classes)) + beta_off_diag
np.fill_diagonal(true_beta, beta_diag)

true_ratio = true_alpha / true_beta

# expected_num_events = (true_mu * end_time) / (1 - true_alpha / true_beta)
# print(expected_num_events)
# exit()

params = {
    'mu': ['mu_mse', 'mu_mse_err'],
    'm': ['ratio_mse', 'ratio_mse_err'],
    'alpha': ['alpha_mse', 'alpha_mse_err'],
    'beta': ['beta_mse', 'beta_mse_err']
}

# known_communities_error
kce = {}
for param, err in params.items():
    kce[err[0]] = []
    kce[err[1]] = []

# estimated_communities_error
ece = copy.deepcopy(kce)
ece['rand_mean'] = []
ece['rand_mean_err'] = []

if run_analysis:
    for j, n_nodes in enumerate(num_nodes_to_test):
        results = Parallel(n_jobs=n_cores)(delayed(calc_mean_and_error_of_count_estiamte)
                                           (n_nodes) for i in range(num_simulations))

        results = np.asarray(results, dtype=np.float)
        results = np.reshape(results, (num_simulations, 9, n_classes, n_classes))

        print(f"Done simulations with {n_nodes} nodes. ARI: {np.mean(results[:, 4, 0, 0])}")

        # estimated communities
        mu_mse_temp = np.power(results[:, 0, :, :] - true_mu, 2)
        ece['mu_mse'].append(np.mean(mu_mse_temp))
        ece['mu_mse_err'].append(2 * np.std(mu_mse_temp) / np.sqrt(mu_mse_temp.size))

        ratio_mse_temp = np.power(results[:, 1, :, :] - true_ratio, 2)
        ece['ratio_mse'].append(np.mean(ratio_mse_temp))
        ece['ratio_mse_err'].append(2 * np.std(ratio_mse_temp) / np.sqrt(ratio_mse_temp.size))

        alpha_mse_temp = np.power(results[:, 2, :, :] - true_alpha, 2)
        ece['alpha_mse'].append(np.mean(alpha_mse_temp))
        ece['alpha_mse_err'].append(2 * np.std(alpha_mse_temp) / np.sqrt(alpha_mse_temp.size))

        beta_mse_temp = np.power(results[:, 3, :, :] - true_beta, 2)
        ece['beta_mse'].append(np.mean(beta_mse_temp))
        ece['beta_mse_err'].append(2 * np.std(beta_mse_temp) / np.sqrt(beta_mse_temp.size))

        rand_mean_temp = np.mean(results[:, 4, 0, 0])
        ece['rand_mean'].append(rand_mean_temp)
        ece['rand_mean_err'].append(2 * np.std(results[:, 4, 0, 0]) / np.sqrt(results[:, 4, 0, 0].size))

        # known communities
        mu_mse_temp = np.power(results[:, 5, :, :] - true_mu, 2)
        kce['mu_mse'].append(np.mean(mu_mse_temp))
        kce['mu_mse_err'].append(2 * np.std(mu_mse_temp) / np.sqrt(mu_mse_temp.size))

        ratio_mse_temp = np.power(results[:, 6, :, :] - true_ratio, 2)
        kce['ratio_mse'].append(np.mean(ratio_mse_temp))
        kce['ratio_mse_err'].append(2 * np.std(ratio_mse_temp) / np.sqrt(ratio_mse_temp.size))

        alpha_mse_temp = np.power(results[:, 7, :, :] - true_alpha, 2)
        kce['alpha_mse'].append(np.mean(alpha_mse_temp))
        kce['alpha_mse_err'].append(2 * np.std(alpha_mse_temp) / np.sqrt(alpha_mse_temp.size))

        beta_mse_temp = np.power(results[:, 8, :, :] - true_beta, 2)
        kce['beta_mse'].append(np.mean(beta_mse_temp))
        kce['beta_mse_err'].append(2 * np.std(beta_mse_temp) / np.sqrt(beta_mse_temp.size))

    with open(f'{result_file_path}/mses.pckl', 'wb') as handle:
        pickle.dump([ece, kce], handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(f'{result_file_path}/mses.pckl', 'rb') as handle:
    ece, kce = pickle.load(handle)


print("Estimated communities:")
print('Rand mean:')
print(ece['rand_mean'])
for param, err in params.items():
    print(param, "MSE:")
    print(ece[err[0]])

print("\nKnown communities:")
for param, err in params.items():
    print(param, "MSE:")
    print(kce[err[0]])


if run_plotting:
    # rand index for estimated communities
    plt.ion()
    plt.subplots(figsize=(3.8, 3))
    plt.bar(range(len(num_nodes_to_test)), ece['rand_mean'], yerr=ece['rand_mean_err'])
    plt.xlabel("Number of Nodes", fontsize=16)
    plt.ylabel("Mean-squared Error", fontsize=16)
    plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"{result_file_path}/plots/estimated_consistent_rand_mean.pdf")

    for param, err in params.items():
        # estimated communities
        mse, mse_err = err[0], err[1]
        plt.ion()
        plt.subplots(figsize=(3.8, 3))
        plt.bar(range(len(num_nodes_to_test)), ece[mse], yerr=ece[mse_err], log=True)
        plt.xlabel("Number of Nodes", fontsize=16)
        plt.ylabel("Mean-squared Error", fontsize=16)
        plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{result_file_path}/plots/estimated_consistent_{param}_mse.pdf")

        # known communities
        plt.ion()
        plt.subplots(figsize=(3.8, 3))
        plt.bar(range(len(num_nodes_to_test)), kce[mse], yerr=kce[mse_err], log=True)
        plt.xlabel("Number of Nodes", fontsize=16)
        plt.ylabel("Mean-squared Error", fontsize=16)
        plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{result_file_path}/plots/known_consistent_{param}_mse.pdf")


if run_regression:
    print("\nRegression: \n")
    print("Estimated Communities:\n")

    start_idx = 0
    end_idx = 3
    num_nodes = num_nodes_to_test[start_idx:end_idx]
    for param, err in params.items():
        print(param, '(estimated communities)')
        x = np.log(num_nodes).reshape(len(num_nodes), 1)
        y = np.log(ece[err[0]][start_idx:end_idx]).reshape(len(ece[err[0]][start_idx:end_idx]), 1)

        reg = LinearRegression().fit(x, y)

        print("R^2:", reg.score(x, y))
        print("coef:", reg.coef_)
        print('intercept:', reg.intercept_)
        print()

        print(param, '(known communities)')
        x = np.log(num_nodes).reshape(len(num_nodes), 1)
        y = np.log(kce[err[0]][start_idx:end_idx]).reshape(len(kce[err[0]][start_idx:end_idx]), 1)

        reg = LinearRegression().fit(x, y)

        print("R^2:", reg.score(x, y))
        print("coef:", reg.coef_)
        print('intercept:', reg.intercept_)
        print()
