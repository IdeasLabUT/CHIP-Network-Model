import numpy as np
import matplotlib.pyplot as plt
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster, regularized_spectral_cluster
from block_generative_model import block_generative_model
from community_generative_model import community_generative_model


# model = "block"
model = "community"

regularized = True

# Setting up model parameters. Mu must be set for the block model. It will be adjusted accordingly.
seed = None
number_of_nodes = 128
class_probabilities = [.25, .25, .25, .25]
durations = [20, 40, 60, 80, 150, 300, 1000]
# durations = [5000, 10000, 25000, 50000]
num_simulation_per_duration = 10

num_of_classes = len(class_probabilities)

bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.8
bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
np.fill_diagonal(bp_mu, 1.8)


if model == "community":
    generative_model = community_generative_model
    bp_mu = utils.get_block_comparable_mu_for_community_model(bp_mu, number_of_nodes, class_probabilities)
else:
    generative_model = block_generative_model

mean_adj_sc_rand_scores = []
mean_adj_sc_rand_scores_err = []

mean_agg_adj_sc_rand_scores = []
mean_agg_adj_sc_rand_scores_err = []

for end_time in durations:
    adj_sc_rands = np.zeros(num_simulation_per_duration)
    agg_adj_sc_rands = np.zeros(num_simulation_per_duration)

    for i in range(num_simulation_per_duration):
        node_membership, event_dicts = generative_model(number_of_nodes,
                                                        class_probabilities,
                                                        bp_mu, bp_alpha, bp_beta,
                                                        end_time, seed=seed)

        true_class_assignments = utils.one_hot_to_class_assignment(node_membership)

        # Spectral clustering on adjacency matrix
        adj = utils.event_dict_to_adjacency(number_of_nodes, event_dicts)

        if regularized:
            adj_sc_pred = regularized_spectral_cluster(adj, num_classes=num_of_classes)
        else:
            adj_sc_pred = spectral_cluster(adj, num_classes=num_of_classes)

        adj_sc_rands[i] = adjusted_rand_score(true_class_assignments, adj_sc_pred)

        # Spectral clustering on aggregated adjacency matrix
        agg_adj = utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts)

        if regularized:
            agg_adj_pred = regularized_spectral_cluster(agg_adj, num_classes=num_of_classes)
        else:
            agg_adj_pred = spectral_cluster(agg_adj, num_classes=num_of_classes)

        agg_adj_sc_rands[i] = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    mean_adj_sc_rand_scores.append(np.mean(adj_sc_rands))
    mean_adj_sc_rand_scores_err.append(2 * np.std(adj_sc_rands) / np.sqrt(len(adj_sc_rands)))

    mean_agg_adj_sc_rand_scores.append(np.mean(agg_adj_sc_rands))
    mean_agg_adj_sc_rand_scores_err.append(2 * np.std(agg_adj_sc_rands) / np.sqrt(len(adj_sc_rands)))

sc_model = "RSC" if regularized else "SC"

print(f"{model} model:")
print("Durations:", durations)
print(f"{sc_model} on Adjacency:", mean_adj_sc_rand_scores)
print(f"{sc_model} on Aggregated Adjacency:", mean_agg_adj_sc_rand_scores)


# Plot Results
fig, ax = plt.subplots()
ind = np.arange(len(durations))    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, mean_adj_sc_rand_scores, width, color='r', yerr=mean_adj_sc_rand_scores_err)
p2 = ax.bar(ind + width, mean_agg_adj_sc_rand_scores, width, color='b', yerr=mean_agg_adj_sc_rand_scores_err)

ax.set_title(f'{model} Model\'s Mean Adjusted Rand Scores')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(durations)
ax.set_ylim(0, 1)

ax.legend((p1[0], p2[0]), (f"{sc_model} on Adjacency", f"{sc_model} on Aggregated Adjacency"))
ax.autoscale_view()
plt.show()
