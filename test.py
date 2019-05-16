import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster, regularized_spectral_cluster
from block_generative_model import block_generative_model
from community_generative_model import community_generative_model


def test_spectral_clustering_on_generative_model(generative_model, regularized,
                                                 n_nodes, n_classes, class_prob,
                                                 mu, alpha, beta, t, seed):

    node_membership, event_dicts = generative_model(n_nodes,
                                                    class_prob,
                                                    mu, alpha, beta,
                                                    t, seed=seed)

    true_class_assignments = utils.one_hot_to_class_assignment(node_membership)

    # Spectral clustering on adjacency matrix
    adj = utils.event_dict_to_adjacency(n_nodes, event_dicts)
    # temp.append(n_nodes ** 2 - np.sum(adj))

    if regularized:
        adj_sc_pred = regularized_spectral_cluster(adj, num_classes=n_classes)
    else:
        adj_sc_pred = spectral_cluster(adj, num_classes=n_classes)

    adj_sc_rand = adjusted_rand_score(true_class_assignments, adj_sc_pred)

    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dicts)
    # utils.plot_degree_count_histogram(agg_adj)

    if regularized:
        agg_adj_pred = regularized_spectral_cluster(agg_adj, num_classes=n_classes)
    else:
        agg_adj_pred = spectral_cluster(agg_adj, num_classes=n_classes)

    agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    return adj_sc_rand, agg_adj_sc_rand


plot_path = '/shared/Results/CommunityHawkes/plots/'

# model = "block"
model = "community"

regularized = False

# Setting up model parameters. Mu must be set for the block model. It will be adjusted accordingly.
seed = None
number_of_nodes_list = [16, 32, 64, 128, 256, 512, 1024]
# number_of_nodes_list = [128]
class_probabilities = [.25, .25, .25, .25]
# durations = [20, 40, 60, 80, 150, 300, 1000]
end_time = 50
num_simulation_per_duration = 100
# num_simulation_per_duration = 1

num_of_classes = len(class_probabilities)

# bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
# bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.8
# bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
# np.fill_diagonal(bp_mu, 1.8)

bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 7500
bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 8000
bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
np.fill_diagonal(bp_mu, 1.8)

if model == "community":
    generative_model = community_generative_model
    bp_mu = utils.scale_parameteres_by_block_pair_size(bp_mu, 128, class_probabilities)
    bp_alpha = utils.scale_parameteres_by_block_pair_size(bp_alpha, 128, class_probabilities)
    bp_beta = utils.scale_parameteres_by_block_pair_size(bp_beta, 128, class_probabilities)
else:
    generative_model = block_generative_model

mean_adj_sc_rand_scores = []
mean_adj_sc_rand_scores_err = []

mean_agg_adj_sc_rand_scores = []
mean_agg_adj_sc_rand_scores_err = []


num_cores = multiprocessing.cpu_count()
for number_of_nodes in number_of_nodes_list:

    results = Parallel(n_jobs=num_cores)(delayed(test_spectral_clustering_on_generative_model)
                                         (generative_model, regularized,
                                          number_of_nodes, num_of_classes, class_probabilities,
                                          bp_mu, bp_alpha, bp_beta, end_time, seed)
                                         for i in range(num_simulation_per_duration))

    # each row is adj_sc_rand, agg_adj_sc_rand
    results = np.asarray(results, dtype=np.float)

    # adj_sc_rands = np.zeros(num_simulation_per_duration)
    # agg_adj_sc_rands = np.zeros(num_simulation_per_duration)
    #
    # for i in range(num_simulation_per_duration):
    #     node_membership, event_dicts = generative_model(number_of_nodes,
    #                                                     class_probabilities,
    #                                                     bp_mu, bp_alpha, bp_beta,
    #                                                     end_time, seed=seed)
    #
    #     true_class_assignments = utils.one_hot_to_class_assignment(node_membership)
    #
    #     # Spectral clustering on adjacency matrix
    #     adj = utils.event_dict_to_adjacency(number_of_nodes, event_dicts)
    #     # temp.append(number_of_nodes ** 2 - np.sum(adj))
    #
    #     if regularized:
    #         adj_sc_pred = regularized_spectral_cluster(adj, num_classes=num_of_classes)
    #     else:
    #         adj_sc_pred = spectral_cluster(adj, num_classes=num_of_classes)
    #
    #     adj_sc_rands[i] = adjusted_rand_score(true_class_assignments, adj_sc_pred)
    #
    #     # Spectral clustering on aggregated adjacency matrix
    #     agg_adj = utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts)
    #     # utils.plot_degree_count_histogram(agg_adj)
    #
    #     if regularized:
    #         agg_adj_pred = regularized_spectral_cluster(agg_adj, num_classes=num_of_classes)
    #     else:
    #         agg_adj_pred = spectral_cluster(agg_adj, num_classes=num_of_classes)
    #
    #     agg_adj_sc_rands[i] = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    mean_adj_sc_rand_scores.append(np.mean(results[:, 0]))
    mean_adj_sc_rand_scores_err.append(2 * np.std(results[:, 0]) / np.sqrt(len(results[:, 0])))

    mean_agg_adj_sc_rand_scores.append(np.mean(results[:, 1]))
    mean_agg_adj_sc_rand_scores_err.append(2 * np.std(results[:, 1]) / np.sqrt(len(results[:, 1])))

sc_model = "RSC" if regularized else "SC"

print(f"{model} model:")
print("Number of nodes:", number_of_nodes_list)
print(f"{sc_model} on Adjacency:", mean_adj_sc_rand_scores)
print(f"{sc_model} on Aggregated Adjacency:", mean_agg_adj_sc_rand_scores)


# Plot Results
fig, ax = plt.subplots()
ind = np.arange(len(number_of_nodes_list))    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, mean_adj_sc_rand_scores, width, color='r', yerr=mean_adj_sc_rand_scores_err)
p2 = ax.bar(ind + width, mean_agg_adj_sc_rand_scores, width, color='b', yerr=mean_agg_adj_sc_rand_scores_err)

ax.set_title(f'{model} Model\'s Mean Adjusted Rand Scores')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(number_of_nodes_list)
ax.set_ylim(0, 1)

ax.legend((p1[0], p2[0]), (f"{sc_model} on Adjacency", f"{sc_model} on Aggregated Adjacency"))
ax.autoscale_view()
plt.savefig(plot_path + "sc-vary.pdf")
plt.show()
