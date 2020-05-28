# -*- coding: utf-8 -*-
"""
"CHIP end-to-end parameter Estimation"

Empirically analyzing the consistency of the CHIP parameter estimators.

@author: Makan Arastuie
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
import model_fitting_utils as model_utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster
from parameter_estimation import estimate_hawkes_from_counts


result_file_path = '/shared/Results/CommunityHawkes/pickles/end_to_end_count_based_estimate/vary_m'

estimate_alpha_beta = True
plot_only = False

no_alpha_name = "_no_alpha" if not estimate_alpha_beta else ""

# sim params
end_time = 100
mu_diag = 1.85
mu_off_diag = 1.65

alpha_diag = 0.75
alpha_off_diag = 0.65

beta_diag = 0.85
beta_off_diag = 0.95

class_probs = [0.25, 0.25, 0.25, 0.25]
num_nodes_to_test = [16, 32, 64, 128, 256, 512]
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

    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
    node_membership = spectral_cluster(agg_adj, num_classes=n_classes, verbose=False)
    sc_rand = adjusted_rand_score(true_node_membership, node_membership)
    sc_rand = np.zeros((n_classes, n_classes)) + sc_rand  # match the shape of other params to retrieve easily

    if estimate_alpha_beta:
        bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio = model_utils.estimate_bp_hawkes_params(event_dict,
                                                                                              node_membership,
                                                                                              end_time,
                                                                                              n_classes)
        return bp_mu, bp_alpha_beta_ratio, bp_alpha, bp_beta, sc_rand

    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
    bp_mu, bp_alpha_beta_ratio = estimate_hawkes_from_counts(agg_adj, node_membership, end_time, 1e-10 / end_time)

    return bp_mu, bp_alpha_beta_ratio, sc_rand


no_alpha_name = "_no_alpha" if not estimate_alpha_beta else ""

true_mu = np.zeros((n_classes, n_classes)) + mu_off_diag
np.fill_diagonal(true_mu, mu_diag)

true_alpha = np.zeros((n_classes, n_classes)) + alpha_off_diag
np.fill_diagonal(true_alpha, alpha_diag)

true_beta = np.zeros((n_classes, n_classes)) + beta_off_diag
np.fill_diagonal(true_beta, beta_diag)

true_ratio = true_alpha / true_beta

# if estimate_alpha_beta:
#     all_estimates = np.zeros((len(num_nodes_to_test), num_simulations, 5, n_classes, n_classes))
# else:
#     all_estimates = np.zeros((len(num_nodes_to_test), num_simulations, 3, n_classes, n_classes))

rand_mean = []
rand_mean_err = []
mu_mse = []
mu_mse_err = []
ratio_mse = []
ratio_mse_err = []

alpha_mse = []
alpha_mse_err = []
beta_mse = []
beta_mse_err = []
if not plot_only:
    for j, n_nodes in enumerate(num_nodes_to_test):
        results = Parallel(n_jobs=n_cores)(delayed(calc_mean_and_error_of_count_estiamte)
                                           (n_nodes) for i in range(num_simulations))

        print(f"Done simulations with {n_nodes} nodes.")

        results = np.asarray(results, dtype=np.float)

        if estimate_alpha_beta:
            results = np.reshape(results, (num_simulations, 5, n_classes, n_classes))
        else:
            results = np.reshape(results, (num_simulations, 3, n_classes, n_classes))

        # all_estimates[j, :, :, :, :] = results

        mu_mse_temp = np.power(results[:, 0, :, :] - true_mu, 2)
        mu_mse.append(np.mean(mu_mse_temp))
        mu_mse_err.append(2 * np.std(mu_mse_temp) / np.sqrt(len(mu_mse_temp)))

        ratio_mse_temp = np.power(results[:, 1, :, :] - true_ratio, 2)
        ratio_mse.append(np.mean(ratio_mse_temp))
        ratio_mse_err.append(2 * np.std(ratio_mse_temp) / np.sqrt(ratio_mse_temp.size))

        rand_mean_temp = np.mean(results[:, 4, 0, 0])
        rand_mean.append(rand_mean_temp)
        rand_mean_err.append(2 * np.std(results[:, 4, 0, 0]) / np.sqrt(results[:, 4, 0, 0].size))

        if estimate_alpha_beta:
            alpha_mse_temp = np.power(results[:, 2, :, :] - true_alpha, 2)
            alpha_mse.append(np.mean(alpha_mse_temp))
            alpha_mse_err.append(2 * np.std(alpha_mse_temp) / np.sqrt(alpha_mse_temp.size))

            beta_mse_temp = np.power(results[:, 3, :, :] - true_beta, 2)
            beta_mse.append(np.mean(beta_mse_temp))
            beta_mse_err.append(2 * np.std(beta_mse_temp) / np.sqrt(beta_mse_temp.size))

    if estimate_alpha_beta:
        with open(f'{result_file_path}/mses.pckl', 'wb') as handle:
            pickle.dump([mu_mse, mu_mse_err, ratio_mse, ratio_mse_err,
                         alpha_mse, alpha_mse_err, beta_mse, beta_mse_err], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{result_file_path}/mses_no_alpha.pckl', 'wb') as handle:
            pickle.dump([mu_mse, mu_mse_err, ratio_mse, ratio_mse_err], handle, protocol=pickle.HIGHEST_PROTOCOL)


if estimate_alpha_beta:
    with open(f'{result_file_path}/mses.pckl', 'rb') as handle:
        [mu_mse, mu_mse_err, ratio_mse, ratio_mse_err,
         alpha_mse, alpha_mse_err, beta_mse, beta_mse_err] = pickle.load(handle)
else:
    with open(f'{result_file_path}/mses_no_alpha.pckl', 'rb') as handle:
        mu_mse, mu_mse_err, ratio_mse, ratio_mse_err = pickle.load(handle)

print("Mu MSE:")
print(rand_mean)

print("Mu MSE:")
print(mu_mse)

print("\nRatio MSE:")
print(ratio_mse)

if estimate_alpha_beta:
    print("\nAlpha MSE:")
    print(alpha_mse)

    print("\nBeta MSE:")
    print(beta_mse)


# plt.title("MSE of Mu estimate using count matrix")
plt.ion()
plt.subplots(figsize=(3.8, 3))
plt.bar(range(len(num_nodes_to_test)), mu_mse, yerr=mu_mse_err, log=True)
plt.xlabel("Number of Nodes", fontsize=16)
plt.ylabel("Mean-squared Error", fontsize=16)
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.tick_params(labelsize=12)
plt.tight_layout()
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)
#plt.autoscale()
plt.savefig(f"{result_file_path}/plots/consistent_mu_mse.pdf")
#plt.show()

#plt.clf()

# plt.title("MSE of m estimate using count matrix")
plt.subplots(figsize=(3.8, 3))
plt.bar(range(len(num_nodes_to_test)), ratio_mse, yerr=ratio_mse_err, log=True)
plt.xlabel("Number of Nodes", fontsize=16)
plt.ylabel("Mean-squared Error", fontsize=16)
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.tick_params(labelsize=12)
plt.tight_layout()
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)
#plt.autoscale()
plt.savefig(f"{result_file_path}/plots/consistent_m_mse.pdf")
#plt.show()

#plt.clf()


plt.subplots(figsize=(3.8, 3))
plt.bar(range(len(num_nodes_to_test)), rand_mean, yerr=rand_mean_err)
# plt.title("MSE of beta estimate using count matrix")
plt.xlabel("Number of Nodes", fontsize=16)
plt.ylabel("Mean Adjusted Rand Score", fontsize=16)

plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.tick_params(labelsize=12)
plt.tight_layout()
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)

plt.savefig(f"{result_file_path}/plots/consistent_rand_mean.pdf")
# plt.show()


if estimate_alpha_beta:
    plt.subplots(figsize=(3.8, 3))
    plt.bar(range(len(num_nodes_to_test)), alpha_mse, yerr=alpha_mse_err, log=True)
    # plt.title("MSE of alpha estimate using count matrix")
    plt.xlabel("Number of Nodes", fontsize=16)
    plt.ylabel("Mean-squared Error", fontsize=16)

    plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)

    plt.savefig(f"{result_file_path}/plots/consistent_alpha_mse.pdf")
    #plt.show()

    #plt.clf()

    plt.subplots(figsize=(3.8, 3))
    plt.bar(range(len(num_nodes_to_test)), beta_mse, yerr=beta_mse_err, log=True)
    # plt.title("MSE of beta estimate using count matrix")
    plt.xlabel("Number of Nodes", fontsize=16)
    plt.ylabel("Mean-squared Error", fontsize=16)

    plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)

    plt.savefig(f"{result_file_path}/plots/consistent_beta_mse.pdf")
    #plt.show()
