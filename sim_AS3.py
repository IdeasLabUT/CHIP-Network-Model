import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


result_file_path = '/shared/Results/CommunityHawkes/pickles/AS3'

plot_only = False

fixed_var = 'n'
# fixed_var = 't'
# fixed_var = 'k'

for fixed_var in ['n', 't', 'k']:
    # Number of test values for all variable must be the same
    n_range = [8, 16, 32, 64, 128, 256, 512]
    t_range = [10, 20, 50, 100, 250, 500, 1000]
    k_range = [15, 10, 8, 5, 3, 2, 1]
    num_test_values = len(n_range)

    fixed_n = 128
    fixed_t = 50
    fixed_k = 4

    num_simulation_per_duration = 100
    n_cores = 34


    def test_spectral_clustering_on_generative_model(n, t, k):
        params = {'number_of_nodes': n,
                  'end_time': t,
                  'class_probabilities': np.ones(k) / k}

        event_dict, true_class_assignments = utils.simulate_community_hawkes(params)

        # Spectral clustering on aggregated adjacency matrix
        agg_adj = utils.event_dict_to_aggregated_adjacency(len(true_class_assignments), event_dict)
        agg_adj_pred = spectral_cluster(agg_adj, num_classes=k)
        agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)

        return agg_adj_sc_rand


    mean_sc_rand_scores = []
    mean_sc_rand_scores_err = []

    n_range_to_test = n_range
    t_range_to_test = t_range
    k_range_to_test = k_range

    if fixed_var == 'n':
        n_range_to_test = [fixed_n]
    elif fixed_var == 't':
        t_range_to_test = [fixed_t]
    else:
        k_range_to_test = [k_range]

    if not plot_only:
        mean_sc_rand_scores = []
        mean_sc_rand_scores_err = []

        cnt = 0
        for n in n_range_to_test:
            for t in t_range_to_test:
                for k in k_range_to_test:
                    results = Parallel(n_jobs=n_cores)(delayed(test_spectral_clustering_on_generative_model)
                                                       (n, t, k) for i in range(num_simulation_per_duration))

                    cnt += 1
                    print(f"Done simulating {cnt} of {num_test_values ** 2}.")

                    results = np.asarray(results, dtype=np.float)

                    mean_sc_rand_scores.append(np.mean(results))
                    mean_sc_rand_scores_err.append(2 * np.std(results) / np.sqrt(len(results)))

        mean_sc_rand_scores = np.reshape(mean_sc_rand_scores, (num_test_values, num_test_values))
        mean_sc_rand_scores_err = np.reshape(mean_sc_rand_scores_err, (num_test_values, num_test_values))

        # Save results
        with open(f'{result_file_path}/all_sims-fixed-{fixed_var}.pckl', 'wb') as handle:
            pickle.dump([mean_sc_rand_scores, mean_sc_rand_scores_err], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{result_file_path}/all_sims-fixed-{fixed_var}.pckl', 'rb') as handle:
        [mean_sc_rand_scores, mean_sc_rand_scores_err] = pickle.load(handle)


    print(f"community model fixed {fixed_var}:")
    print(f"rand:", mean_sc_rand_scores)
    print(f"rand error:", mean_sc_rand_scores_err)


    # # Plot Results
    # fig, ax = plt.subplots()
    #
    # if fixed_var == 'n':
    #     ylables = t_range
    #     xlables = k_range
    #     plt.ylabel("T")
    #     plt.xlabel("K")
    #
    # elif fixed_var == 't':
    #     ylables = n_range
    #     xlables = k_range
    #     plt.ylabel("N")
    #     plt.xlabel("K")
    #
    # else:
    #     ylables = n_range
    #     xlables = t_range
    #     plt.ylabel("N")
    #     plt.xlabel("T")
    #
    # im, _ = heatmap(mean_sc_rand_scores, ylables, xlables, ax=ax, cmap="Greys", cbarlabel=f"Fixed {fixed_var}")
    #
    # ax.set_title(f"AS3 Fixed {fixed_var} Community Model SC Rand")
    # fig.tight_layout()
    # plt.show()
