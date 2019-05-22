import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from parameter_estimation import estimate_hawkes_from_counts
from community_generative_model import community_generative_model


def calc_mean_and_error_of_count_estiamte(n_nodes, class_probabilities, bp_mu, bp_alpha, bp_beta, burnin,
                                          end_time, seed):

    node_membership, event_dicts = community_generative_model(n_nodes,
                                                              class_probabilities,
                                                              bp_mu, bp_alpha, bp_beta,
                                                              burnin=burnin, end_time=end_time, seed=seed)

    node_membership = utils.one_hot_to_class_assignment(node_membership)
    event_agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dicts, dtype=np.int)

    return estimate_hawkes_from_counts(event_agg_adj, node_membership, end_time)


seed = None
num_simulations = 100
num_nodes_to_test = [4, 8, 16, 32, 64, 128, 256, 512]
end_time = 100

class_probabilities = [1]
num_of_classes = len(class_probabilities)

bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * .7500
bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * .8000
bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
np.fill_diagonal(bp_mu, 1.8)

# bp_mu = utils.scale_parameteres_by_block_pair_size(bp_mu, 128, class_probabilities)
# bp_alpha = utils.scale_parameteres_by_block_pair_size(bp_alpha, 128, class_probabilities)
# bp_beta = utils.scale_parameteres_by_block_pair_size(bp_beta, 128, class_probabilities)

true_ratio = bp_alpha[0, 0]/bp_beta[0, 0]
true_mu = bp_mu[0, 0]

mu_mse = []
ratio_mse = []

for n_nodes in num_nodes_to_test:
    results = Parallel(n_jobs=20)(delayed(calc_mean_and_error_of_count_estiamte)
                                         (n_nodes, class_probabilities,
                                          bp_mu, bp_alpha, bp_beta,
                                          burnin=None, end_time=end_time, seed=seed)
                                          for i in range(num_simulations))

    print(f"Done simulations with {n_nodes} nodes.")

    # each row is mu, alpha_beta_ratio
    results = np.asarray(results, dtype=np.float)
    results = np.reshape(results, (num_simulations, 2))

    mu_mse.append(np.mean(np.power(results[:, 0] - true_mu, 2)))
    ratio_mse.append(np.mean(np.power(results[:, 1] - true_ratio, 2)))

print(mu_mse)
print(ratio_mse)

plt.bar(range(len(num_nodes_to_test)), mu_mse)
plt.title("MSE of Mu estimate using count matrix")
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
