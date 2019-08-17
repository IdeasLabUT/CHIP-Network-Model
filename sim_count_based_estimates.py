# -*- coding: utf-8 -*-
"""
"Hawkes Process Parameter Estimation"

Empirically analyzing the consistency of the CHIP parameter estimators.

@author: Makan Arastuie
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
import model_fitting_utils as model_utils
from parameter_estimation import estimate_hawkes_from_counts


result_file_path = './storage/results/count_based_estimate'

estimate_alpha_beta = True
plot_only = True

no_alpha_name = "_no_alpha" if not estimate_alpha_beta else ""

# sim params
end_time = 100
alpha = 7.
beta = 8.
mu_diag = 1.8
class_probs = [1]

num_nodes_to_test = [4, 8, 16, 32, 64, 128, 256]
num_simulations = 100
n_cores = 6


def calc_mean_and_error_of_count_estiamte(n_nodes):
    params = {'number_of_nodes': n_nodes,
              'class_probabilities': class_probs,
              'end_time': end_time,
              'alpha': alpha,
              'beta': beta,
              'mu_diag': mu_diag,
              'scale': False}

    event_dict, node_membership = utils.simulate_community_hawkes(params)

    if estimate_alpha_beta:
        bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio = model_utils.estimate_bp_hawkes_params(event_dict,
                                                                                              node_membership,
                                                                                              end_time,
                                                                                              len(class_probs))
        return bp_mu, bp_alpha_beta_ratio, bp_alpha, bp_beta

    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
    bp_mu, bp_alpha_beta_ratio = estimate_hawkes_from_counts(agg_adj, node_membership, end_time, 1e-10 / end_time)

    return bp_mu, bp_alpha_beta_ratio


no_alpha_name = "_no_alpha" if not estimate_alpha_beta else ""

true_ratio = alpha/beta
true_mu = mu_diag

if estimate_alpha_beta:
    all_estimates = np.zeros((len(num_nodes_to_test), num_simulations, 4))
else:
    all_estimates = np.zeros((len(num_nodes_to_test), num_simulations, 2))

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
            results = np.reshape(results, (num_simulations, 4))
        else:
            results = np.reshape(results, (num_simulations, 2))

        all_estimates[j, :, :] = results

        mu_mse_temp = np.power(results[:, 0] - true_mu, 2)
        mu_mse.append(np.mean(mu_mse_temp))
        mu_mse_err.append(2 * np.std(mu_mse_temp) / np.sqrt(len(mu_mse_temp)))

        ratio_mse_temp = np.power(results[:, 1] - true_ratio, 2)
        ratio_mse.append(np.mean(ratio_mse_temp))
        ratio_mse_err.append(2 * np.std(ratio_mse_temp) / np.sqrt(len(ratio_mse_temp)))

        if estimate_alpha_beta:
            alpha_mse_temp = np.power(results[:, 2] - alpha, 2)
            alpha_mse.append(np.mean(alpha_mse_temp))
            alpha_mse_err.append(2 * np.std(alpha_mse_temp) / np.sqrt(len(alpha_mse_temp)))

            beta_mse_temp = np.power(results[:, 3] - beta, 2)
            beta_mse.append(np.mean(beta_mse_temp))
            beta_mse_err.append(2 * np.std(beta_mse_temp) / np.sqrt(len(beta_mse_temp)))

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
print(mu_mse)

print("\nRatio MSE:")
print(ratio_mse)

if estimate_alpha_beta:
    print("\nAlpha MSE:")
    print(alpha_mse)

    print("\nBeta MSE:")
    print(beta_mse)


# plt.title("MSE of Mu estimate using count matrix")
plt.bar(range(len(num_nodes_to_test)), mu_mse, yerr=mu_mse_err)
plt.xlabel("Number of Nodes", fontsize=16)
plt.ylabel("Mean Squared Error", fontsize=16)
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.tick_params(labelsize=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)
plt.autoscale()
plt.savefig(f"{result_file_path}/plots/mu_mse.pdf")
plt.show()

plt.clf()

# plt.title("MSE of m estimate using count matrix")
plt.bar(range(len(num_nodes_to_test)), ratio_mse, yerr=ratio_mse_err)
plt.xlabel("Number of Nodes", fontsize=16)
plt.ylabel("Mean Squared Error", fontsize=16)
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.tick_params(labelsize=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)
plt.autoscale()
plt.savefig(f"{result_file_path}/plots/m_mse.pdf")
plt.show()

plt.clf()

if estimate_alpha_beta:
    plt.bar(range(len(num_nodes_to_test)), alpha_mse, yerr=alpha_mse_err)
    # plt.title("MSE of alpha estimate using count matrix")
    plt.xlabel("Number of Nodes", fontsize=16)
    plt.ylabel("Mean Squared Error", fontsize=16)

    plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
    plt.tick_params(labelsize=12)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)

    plt.savefig(f"{result_file_path}/plots/alpha_mse.pdf")
    plt.show()

    plt.clf()

    plt.bar(range(len(num_nodes_to_test)), beta_mse, yerr=beta_mse_err)
    # plt.title("MSE of beta estimate using count matrix")
    plt.xlabel("Number of Nodes", fontsize=16)
    plt.ylabel("Mean Squared Error", fontsize=16)

    plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
    plt.tick_params(labelsize=12)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), fontsize=16)

    plt.savefig(f"{result_file_path}/plots/beta_mse.pdf")
    plt.show()
