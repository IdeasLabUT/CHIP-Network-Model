import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster


'''
** simulation AS2: fix $n,k,T$, then:

(a) Increase mu_diag and mu_off_diag such that the ratio mu_off_diag/mu_diag remains the same.
(b) Hold mu_off_diag fixed and only increase mu_diag slowly. 

Expectation: We should see accuracy increase in both these cases. When mu_diag/mu_off_diag ratio is low, the algorithms 
will do poorly, but as the ratio increases there is more signal and the algorithm will do well and go all the way to 1.
'''

result_file_path = '/shared/Results/CommunityHawkes/pickles/AS2'

sim_type = 'a'
# sim_type = 'b'

plot_only = False

plot_name = "fixed_ratio" if sim_type == 'a' else "increase_mu_diag"

a_scalars_to_test = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# a_scalars_to_test = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
b_scalars_to_test = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]

n_classes = 4
num_simulation_per_duration = 100
n_cores = 34

scalars_to_test = a_scalars_to_test if sim_type == 'a' else b_scalars_to_test


def test_spectral_clustering_on_generative_model(scalar):
    params = {'alpha': 0.05,
              'beta': 0.08,
              'mu_diag': 0.075 * scalar,
              'mu_off_diag': 0.065 if sim_type == 'b' else 0.065 * scalar,
              'scale': False}

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params)

    # Spectral clustering on aggregated adjacency matrix
    agg_adj = utils.event_dict_to_aggregated_adjacency(len(true_class_assignments), event_dict)
    agg_adj_pred = spectral_cluster(agg_adj, num_classes=n_classes)
    agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)

    return agg_adj_sc_rand


if not plot_only:
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
p1 = ax.bar(ind, mean_sc_rand_scores, color='c', yerr=mean_sc_rand_scores_err)

# ax.set_title(f'AS2 {sim_type} Community Model\'s Mean Adjusted Rand Scores')
ax.set_xticks(ind)
ax.tick_params(labelsize=12)
ax.set_xticklabels(scalars_to_test, fontsize=12)
plt.xlabel("Scalars", fontsize=16)
plt.ylabel("Mean Adjusted Rand Score", fontsize=16)

ax.set_ylim(0, 1)

ax.autoscale_view()

plt.savefig(result_file_path + "/plots/" + plot_name + ".pdf")
plt.show()
