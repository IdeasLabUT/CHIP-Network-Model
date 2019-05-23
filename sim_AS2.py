import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster

result_file_path = '/shared/Results/CommunityHawkes/pickles/AS2'

sim_type = 'a'
# sim_type = 'b'

scalars_to_test = [0.001, 0.01, 0.05, 0.1, 1, 10, 25, 50, 100, 200]
n_classes = 4
num_simulation_per_duration = 100
n_cores = 34


def test_spectral_clustering_on_generative_model(scalar):
    params = {'mu_off_diag': 0.6 * scalar,
              'mu_diag': 1.8 if sim_type == 'b' else 1.8 * scalar,
              'scale': True}

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params)

    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(len(true_class_assignments), event_dict)
    agg_adj_pred = spectral_cluster(agg_adj, num_classes=n_classes)
    agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    return agg_adj_sc_rand


mean_sc_rand_scores = []
mean_sc_rand_scores_err = []

for scalar in scalars_to_test:

    results = Parallel(n_jobs=n_cores)(delayed(test_spectral_clustering_on_generative_model)
                                       (scalar) for i in range(num_simulation_per_duration))

    print(f"Done simulating with {scalar} scalar.")

    results = np.asarray(results, dtype=np.float)

    mean_sc_rand_scores.append(np.mean(results))
    mean_sc_rand_scores_err.append(2 * np.std(results) / np.sqrt(len(results)))


# Save results
with open(f'{result_file_path}/all_sims-{sim_type}.pckl', 'wb') as handle:
    pickle.dump([mean_sc_rand_scores, mean_sc_rand_scores_err], handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{result_file_path}/all_sims-{sim_type}.pckl', 'rb') as handle:
    [mean_sc_rand_scores, mean_sc_rand_scores_err] = pickle.load(handle)


print(f"community model:")
print("Number of nodes:", scalars_to_test)
print(f"rand:", mean_sc_rand_scores)
print(f"rand error:", mean_sc_rand_scores_err)


# Plot Results
fig, ax = plt.subplots()
ind = np.arange(len(scalars_to_test))    # the x locations for the groups
p1 = ax.bar(ind, mean_sc_rand_scores, color='r', yerr=mean_sc_rand_scores_err)

ax.set_title(f'AS2 {sim_type} Community Model\'s Mean Adjusted Rand Scores')
ax.set_xticks(ind)
ax.set_xticklabels(scalars_to_test)
ax.set_ylim(0, 1)
ax.autoscale_view()
# plt.savefig(plot_path + "sc-vary.pdf")
plt.show()
