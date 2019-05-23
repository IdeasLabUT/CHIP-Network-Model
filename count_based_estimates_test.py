import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
import community_model_fitting_utils as model_utils

result_file_path = '/shared/Results/CommunityHawkes/pickles/count_based_estimate'

# sim params
end_time = 100
alpha = 2.75
beta = 2.8
mu_diag = 3.8
class_probs = [1]

num_nodes_to_test = [4, 8, 16, 32, 64, 128, 256]
num_simulations = 100
n_cores = 5


def calc_mean_and_error_of_count_estiamte(n_nodes):
    params = {'number_of_nodes': n_nodes,
              'class_probabilities': class_probs,
              'end_time': end_time,
              'alpha': alpha,
              'beta': beta,
              'mu_diag': mu_diag,
              'scale': False}

    event_dict, node_membership = utils.simulate_community_hawkes(params)

    bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio = model_utils.estimate_bp_hawkes_params(event_dict, node_membership,
                                                                                          end_time, len(class_probs))

    return bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio

true_ratio = alpha/beta
true_mu = mu_diag

all_estimates = np.zeros((len(num_nodes_to_test), num_simulations, 4))

mu_mse = []
alpha_mse = []
beta_mse = []
ratio_mse = []

for j, n_nodes in enumerate(num_nodes_to_test):
    results = Parallel(n_jobs=n_cores)(delayed(calc_mean_and_error_of_count_estiamte)
                                       (n_nodes) for i in range(num_simulations))

    print(f"Done simulations with {n_nodes} nodes.")

    # each row is mu, alpha_beta_ratio
    results = np.asarray(results, dtype=np.float)
    results = np.reshape(results, (num_simulations, 4))

    all_estimates[j, :, :] = results

    mu_mse.append(np.mean(np.power(results[:, 0] - true_mu, 2)))
    alpha_mse.append(np.mean(np.power(results[:, 1] - alpha, 2)))
    beta_mse.append(np.mean(np.power(results[:, 2] - beta, 2)))
    ratio_mse.append(np.mean(np.power(results[:, 3] - true_ratio, 2)))


# Save results
with open(f'{result_file_path}/all_estimates.pckl', 'wb') as handle:
    pickle.dump(all_estimates, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{result_file_path}/all_estimates.pckl', 'rb') as handle:
    all_estimates = pickle.load(handle)


with open(f'{result_file_path}/mses.pckl', 'wb') as handle:
    pickle.dump([mu_mse, alpha_mse, beta_mse, ratio_mse], handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{result_file_path}/mses.pckl', 'rb') as handle:
    mu_mse, alpha_mse, beta_mse, ratio_mse = pickle.load(handle)


print("Mu MSE:")
print(mu_mse)

print("\nAlpha MSE:")
print(alpha_mse)

print("\nBeta MSE:")
print(beta_mse)

print("\nRatio MSE:")
print(ratio_mse)


plt.bar(range(len(num_nodes_to_test)), mu_mse)
plt.title("MSE of Mu estimate using count matrix")
plt.xlabel("Number of Nodes")
plt.ylabel("Mean Squared Error")
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.show()

plt.bar(range(len(num_nodes_to_test)), alpha_mse)
plt.title("MSE of alpha estimate using count matrix")
plt.xlabel("Number of Nodes")
plt.ylabel("Mean Squared Error")
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.show()

plt.bar(range(len(num_nodes_to_test)), beta_mse)
plt.title("MSE of beta estimate using count matrix")
plt.xlabel("Number of Nodes")
plt.ylabel("Mean Squared Error")
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.show()

plt.bar(range(len(num_nodes_to_test)), ratio_mse)
plt.title("MSE of m estimate using count matrix")
plt.xlabel("Number of Nodes")
plt.ylabel("Mean Squared Error")
plt.xticks(range(len(num_nodes_to_test)), num_nodes_to_test)
plt.show()
