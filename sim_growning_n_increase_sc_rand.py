import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster

result_file_path = '/shared/Results/CommunityHawkes/pickles/growing_n_increase_sc_rand'

agg_adj_should_fail = True
plot_only = True

number_of_nodes_list = [8, 16, 32, 64, 128, 256, 512]
# number_of_nodes_list = [8, 16, 32, 64, 128]
n_classes = 4
num_simulation_per_duration = 30
n_cores = 30


def test_spectral_clustering_on_generative_model(n_nodes):
    if agg_adj_should_fail:
        params = {'number_of_nodes': n_nodes,
                  'alpha': 7.0,
                  'beta': 8.0,
                  'mu_off_diag': 0.001,
                  'mu_diag': 0.002,
                  'scale': False,
                  'end_time': 400}
    else:
        params = {'number_of_nodes': n_nodes,
                  'alpha': 0.001,
                  'beta': 0.008,
                  'mu_off_diag': 0.001,
                  'mu_diag': 0.001,
                  'alpha_diag': 0.006,
                  'scale': False,
                  'end_time': 400}

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params)

    # Spectral clustering on adjacency matrix
    adj = utils.event_dict_to_adjacency(n_nodes, event_dict)
    adj_sc_pred = spectral_cluster(adj, num_classes=n_classes)
    adj_sc_rand = adjusted_rand_score(true_class_assignments, adj_sc_pred)

    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
    agg_adj_pred = spectral_cluster(agg_adj, num_classes=n_classes)
    agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    return adj_sc_rand, agg_adj_sc_rand


# Save results
file_name = "all_sims_agg_adj_fail.pckl" if agg_adj_should_fail else "all_sims_adj_fail.pckl"

if not plot_only:
    mean_adj_sc_rand_scores = []
    mean_adj_sc_rand_scores_err = []

    mean_agg_adj_sc_rand_scores = []
    mean_agg_adj_sc_rand_scores_err = []

    for number_of_nodes in number_of_nodes_list:

        results = Parallel(n_jobs=n_cores)(delayed(test_spectral_clustering_on_generative_model)
                                           (number_of_nodes)
                                           for i in range(num_simulation_per_duration))

        print(f"Done simulating with {number_of_nodes} nodes.")

        # each row is adj_sc_rand, agg_adj_sc_rand
        results = np.asarray(results, dtype=np.float)

        mean_adj_sc_rand_scores.append(np.mean(results[:, 0]))
        mean_adj_sc_rand_scores_err.append(2 * np.std(results[:, 0]) / np.sqrt(len(results[:, 0])))

        mean_agg_adj_sc_rand_scores.append(np.mean(results[:, 1]))
        mean_agg_adj_sc_rand_scores_err.append(2 * np.std(results[:, 1]) / np.sqrt(len(results[:, 1])))

    with open(f'{result_file_path}/{file_name}', 'wb') as handle:
        pickle.dump([mean_adj_sc_rand_scores,
                     mean_adj_sc_rand_scores_err,
                     mean_agg_adj_sc_rand_scores,
                     mean_agg_adj_sc_rand_scores_err], handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{result_file_path}/{file_name}', 'rb') as handle:
    [mean_adj_sc_rand_scores,
     mean_adj_sc_rand_scores_err,
     mean_agg_adj_sc_rand_scores,
     mean_agg_adj_sc_rand_scores_err] = pickle.load(handle)


print(f"community model:")
print("Number of nodes:", number_of_nodes_list)
print(f"SC on Adjacency:", mean_adj_sc_rand_scores)
print(f"SC on Aggregated Adjacency:", mean_agg_adj_sc_rand_scores)


# Plot Results
fig, ax = plt.subplots()
ind = np.arange(len(number_of_nodes_list))    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, mean_adj_sc_rand_scores, width, color='r', yerr=mean_adj_sc_rand_scores_err)
p2 = ax.bar(ind + width, mean_agg_adj_sc_rand_scores, width, color='b', yerr=mean_agg_adj_sc_rand_scores_err)

# ax.set_title(f'Community Model\'s Mean Adjusted Rand Scores')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(number_of_nodes_list, fontsize=12)
ax.set_ylim(0, 1)
ax.tick_params(labelsize=12)

ax.legend((p1[0], p2[0]), (f"Unweighted Adjacency", f"Weighted Adjacency"), fontsize=14)
plt.ylabel("Mean Adjusted Rand Score", fontsize=16)
plt.xlabel("Number of Nodes", fontsize=16)

ax.autoscale_view()

plot_name = "agg_adj_fail" if agg_adj_should_fail else "adj_fail"
plt.savefig(f"{result_file_path}/plots/{plot_name}.pdf")
# plt.show()