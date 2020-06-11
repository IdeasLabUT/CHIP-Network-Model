# -*- coding: utf-8 -*-
"""
"Exploratory Analysis of the Facebook Wall Posts Dataset using CHIP"

Here we fit CHIP to the largest connected component of the Facebook wall-post dataset and analyze various aspects of
the fitted model.

@author: Makan Arastuie
"""

import time
import pickle
import numpy as np
import dataset_utils
import matplotlib.pyplot as plt
from plotting_utils import heatmap
import generative_model_utils as utils
import model_fitting_utils as fitting_utils
import parameter_estimation as estimate_utils
from spectral_clustering import spectral_cluster

result_file_path = '/shared/Results/CommunityHawkes/pickles/fb_chip_fit_2'

fit_chip = False
load_fb = True
plot_hawkes_params = False
plot_node_membership = False
plot_num_events = False
plot_community_structure = True
simulate_chip = False
get_confidence_intervals = False
verbose = False
num_classes = 10

# load Facebook Wall-posts
if fit_chip or load_fb or simulate_chip or plot_num_events or get_confidence_intervals:
    tic = time.time()
    fb_event_dict, fb_num_node, fb_duration = dataset_utils.load_facebook_wall(largest_connected_component_only=True)
    toc = time.time()

    print(f"Loaded the dataset in {toc - tic:.1f}s")

    num_events = utils.num_events_in_event_dict(fb_event_dict)
    if verbose:
        print("Num Nodes:", fb_num_node, "Duration:", fb_duration,
              "Num Edges:", num_events)

# fit Facebook Wall-posts
if fit_chip:
    tic = time.time()
    agg_adj = utils.event_dict_to_aggregated_adjacency(fb_num_node, fb_event_dict)
    adj = utils.event_dict_to_adjacency(fb_num_node, fb_event_dict)
    toc = time.time()

    if verbose:
        print(f"Generated aggregated adj in {toc - tic:.1f}s")

    tic_tot = time.time()
    tic = time.time()
    # Running spectral clustering
    node_membership = spectral_cluster(agg_adj, num_classes=num_classes, verbose=False, plot_eigenvalues=False)

    toc = time.time()

    print(f"Spectral clustering done in {toc - tic:.1f}s")

    if verbose:
        print("Community assignment prob:", np.unique(node_membership, return_counts=True)[1] / fb_num_node)

    tic = time.time()
    bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                            fb_duration,
                                                                            1e-10 / fb_duration)
    toc = time.time()

    print(f"Mu and m estimated in {toc - tic:.1f}s")

    if verbose:
        print("Mu:")
        print(bp_mu)
        print("Ratio:")
        print(bp_alpha_beta_ratio)

    print("\nStart Beta estimation:")

    tic = time.time()
    bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)
    block_pair_events = utils.event_dict_to_block_pair_events(fb_event_dict, node_membership, num_classes)
    bp_size = utils.calc_block_pair_size(node_membership, num_classes)

    cnt = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
                                                                            bp_mu[b_i, b_j],
                                                                            bp_alpha_beta_ratio[b_i, b_j],
                                                                            fb_duration, bp_size[b_i, b_j])
            cnt += 1
            print(f"{100 * cnt / num_classes ** 2:0.2f}% Done.", end='\r')

    bp_alpha = bp_alpha_beta_ratio * bp_beta
    toc = time.time()
    toc_tot = time.time()

    print(f"Beta estimated in {toc - tic:.1f}s")
    print(f"Total computation time: {toc_tot - tic_tot:.1f}s")

    if verbose:
        print("Alpha")
        print(bp_alpha)

        print("Beta")
        print(bp_beta)

    hawkes_params = {'mu': bp_mu,
                     'alpha': bp_alpha,
                     'beta': bp_beta,
                     'alpha_beta_ratio': bp_alpha_beta_ratio}
    # Save results
    with open(f'{result_file_path}/all_model_params-k-{num_classes}.pckl', 'wb') as handle:
        pickle.dump([fb_num_node, num_events, fb_duration,
                     node_membership,
                     hawkes_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{result_file_path}/all_model_params-k-{num_classes}.pckl', 'rb') as handle:
        [fb_num_node, num_events, fb_duration,
         node_membership,
         hawkes_params] = pickle.load(handle)

print("Diag mu mean: ", np.mean(hawkes_params['mu'][np.eye(num_classes, dtype=bool)]))
print("off-diag mu mean: ", np.mean(hawkes_params['mu'][~np.eye(num_classes, dtype=bool)]))

if plot_num_events:
    block_pair_events = utils.event_dict_to_block_pair_events(fb_event_dict, node_membership, num_classes)

    # plot number of events per block pair
    num_events_block_pair = np.zeros((num_classes, num_classes), dtype=np.int)
    for i in range(num_classes):
        for j in range(num_classes):
            num_events_block_pair[i, j] = len(np.concatenate(block_pair_events[i][j]))

    fig, ax = plt.subplots()
    labels = np.arange(1, num_classes + 1)
    im, _ = heatmap(num_events_block_pair, labels, labels, ax=ax, cmap="Greys", cbarlabel=" ")

    fig.tight_layout()
    plt.savefig(f"{result_file_path}/plots/num_block_pair_events.pdf")
    # plt.show()

    bp_size = utils.calc_block_pair_size(node_membership, num_classes)
    mean_num_events_block_pair = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            mean_num_events_block_pair[i, j] = len(np.concatenate(block_pair_events[i][j])) / bp_size[i, j]

    fig, ax = plt.subplots()
    labels = np.arange(1, num_classes + 1)
    im, _ = heatmap(mean_num_events_block_pair, labels, labels, ax=ax, cmap="Greys",
                    cbarlabel=" ")

    fig.tight_layout()
    plt.savefig(f"{result_file_path}/plots/mean_num_node_pair_event.pdf")
    # plt.show()

if plot_node_membership:
    # # Node membership percentage
    blocks, counts = np.unique(node_membership, return_counts=True)
    percent_membership = 100 * counts / np.sum(counts)
    fig, ax = plt.subplots()
    ind = np.arange(1, 11)    # the x locations for the groups
    width = 0.75
    p1 = ax.bar(ind, percent_membership, width, color='blue')

    rects = ax.patches
    for rect, label in zip(rects, counts):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,
                ha='center', va='bottom', rotation='vertical', fontsize=12)

    ax.set_xticks(ind)
    ax.tick_params(labelsize=12)
    ax.set_xticklabels(np.arange(1, 11), fontsize=12)
    plt.xlabel("Blocks", fontsize=16)
    plt.ylabel("Percentage of Total Population", fontsize=16)
    ax.set_ylim(0, 25)
    ax.autoscale_view()
    plt.savefig(f"{result_file_path}/plots/block_size.pdf")
    # plt.show()

# Plot Results
if plot_hawkes_params:
    cbar_label = {
        'mu': r'$\mu$',
        'alpha': r'$\alpha$',
        'beta': r'$\beta$'
    }

    for param in ['mu', 'alpha', 'beta']:
        fig, ax = plt.subplots()
        labels = np.arange(1, num_classes + 1)
        im, _ = heatmap(hawkes_params[param], labels, labels, ax=ax, cmap="Greys", cbarlabel=cbar_label[param])

        ax.set_title(f"Full Facebook wall-posts {param.capitalize()}")
        fig.tight_layout()
        # plt.savefig(f"{result_file_path}/plots/{param}-k-{num_classes}.pdf")
        plt.show()

    # plot m
    fig, ax = plt.subplots()
    labels = np.arange(1, num_classes + 1)
    im, _ = heatmap(hawkes_params['alpha'] / hawkes_params['beta'], labels, labels, ax=ax, cmap="Greys",
                    cbarlabel=r"$m$")

    fig.tight_layout()
    plt.show()
    plt.savefig(f"{result_file_path}/plots/m-k-{num_classes}.pdf")

    # plot mu / (1 - m)
    fig, ax = plt.subplots()
    labels = np.arange(1, num_classes + 1)
    m = hawkes_params['alpha'] / hawkes_params['beta']
    expected = hawkes_params['mu'] / (1 - m)
    im, _ = heatmap(expected, labels, labels, ax=ax, cmap="Greys",
                    cbarlabel=r"$\mu/(1-m)$")

    fig.tight_layout()
    plt.show()
    plt.savefig(f"{result_file_path}/plots/mu-over-1-m-k-{num_classes}.pdf")


if get_confidence_intervals:
    mu_ci, m_ci = fitting_utils.compute_mu_and_m_confidence_interval(fb_event_dict, node_membership, num_classes,
                                                                     z_alpha=0.05, duration=fb_duration)
    print("m CI:")
    print('[', end='')
    for i in range(num_classes):
        print('[', end='')
        for j in range(num_classes):
            print(f"{hawkes_params['alpha_beta_ratio'][i, j]:.3f} +/- {m_ci[i, j]:.3f}", end=', ')
        print(']')
    print(']')

    print("mu CI:")
    print('[', end='')
    for i in range(num_classes):
        print('[', end='')
        for j in range(num_classes):
            print(f"{hawkes_params['mu'][i, j]:.2e} +/- {mu_ci[i, j]:.2e}", end=', ')
        print(']')
    print(']')

    # set the tuple list to the pairs that are important
    block_pair_tuple_list = [(0, 0, 0, 1), (0, 0, 1, 0), (1, 1, 0, 1), (1, 1, 1, 0)]
    mu_pairwise_diff_res = fitting_utils.compute_mu_pairwise_difference_confidence_interval(fb_event_dict,
                                                                                            node_membership,
                                                                                            num_classes,
                                                                                            hawkes_params['mu'],
                                                                                            fb_duration,
                                                                                            block_pair_tuple_list,
                                                                                            z_alpha=0.05)
    print("mu pairwise CI:", mu_pairwise_diff_res)


if simulate_chip:
    # # Generating a CHIP model with fitted parameters
    # [generated_node_membership,
    #  generated_event_dict] = fitting_utils.generate_fit_community_hawkes(fb_event_dict, node_membership,
    #                                                                      hawkes_params['mu'], hawkes_params['alpha'],
    #                                                                      hawkes_params['beta'], fb_duration,
    #                                                                      plot_hist=False, n_cores=25)
    #
    # with open(f'{result_file_path}/generated_model.pckl', 'wb') as handle:
    #     pickle.dump([generated_node_membership, generated_event_dict], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Loading a CHIP model with fitted parameters
    with open(f'{result_file_path}/generated_model.pckl', 'rb') as handle:
        generated_node_membership, generated_event_dict = pickle.load(handle)

    # Generating a CHIP model with fitted parameters
    generated_agg_adj = utils.event_dict_to_aggregated_adjacency(fb_num_node, generated_event_dict, dtype=np.int)
    generated_deg_count_flattened = np.reshape(generated_agg_adj, (fb_num_node * fb_num_node))

    fb_agg_adj = utils.event_dict_to_aggregated_adjacency(fb_num_node, fb_event_dict, dtype=np.int)
    deg_count_flattened = np.reshape(fb_agg_adj, (fb_num_node * fb_num_node))

    # plt.hist(deg_count_flattened, bins=50, alpha=0.5, label='Real Data', color='blue', density=True)
    # plt.hist(generated_deg_count_flattened, bins=25, alpha=0.5, label='Generated Data', color='red', density=True)
    #
    # plt.legend(loc='upper right')
    # plt.xlabel('Event Count')
    # plt.ylabel('Density')
    # plt.title(f'Histogram of the Count Matrix Real Vs. Generated CHIP Model Data - K: {num_classes}'
    #           f'\n Mean Count -  Real: {np.mean(fb_agg_adj):.3f} - Generated: {np.mean(generated_agg_adj):.3f}')
    # plt.yscale("log")
    # plt.show()
    #
    # plt.clf()

    # out degree -> axis=1, in degree -> axis=0
    fb_degree = np.sum(fb_agg_adj, axis=1) + 1
    fb_deg_count = np.unique(fb_degree, return_counts=True)
    fb_deg_count = fitting_utils.log_binning(fb_deg_count, 75)

    gen_degree = np.sum(generated_agg_adj, axis=1) + 1
    gen_deg_count = np.unique(gen_degree, return_counts=True)
    gen_deg_count = fitting_utils.log_binning(gen_deg_count, 75)

    plt.xscale('log')
    plt.yscale('log')

    plt.scatter(fb_deg_count[0], fb_deg_count[1], c='b', marker='*', alpha=0.9, label="Real")
    plt.scatter(gen_deg_count[0], gen_deg_count[1], c='r', marker='x', alpha=0.9, label="Generated")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="best", fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xlabel('Degree', fontsize=20)
    plt.tight_layout()
    plt.title(f'Out Degree Distribution on Log-Log Scale')
    # plt.savefig('{}degree-dist-{}.pdf'.format(plot_save_path, pymk_directories[i]), format='pdf')
    plt.show()
    # plt.clf()


# Loaded the dataset in 6.8s
# (43953, 43953) 100
# Spectral clustering done in 68.0s
# Mu and m estimated in 12.5s
#
# Start Beta estimation:
# Beta estimated in 88.2s
# Total computation time: 168.6s


if plot_community_structure:
    # adj = utils.event_dict_to_adjacency(fb_num_node, fb_event_dict, dtype=np.int)
    num_nodes = len(node_membership)
    community_membership = utils.node_membership_to_community_membership(node_membership, num_classes)
    community_size = [len(community) for community in community_membership]
    node_ids = np.concatenate(community_membership)
    sorting_map = {}
    for i in range(node_ids.shape[0]):
        sorting_map[node_ids[i]] = i

    sorted_adj = np.zeros((num_nodes, num_nodes), dtype=np.int)

    for (u, v), event_times in fb_event_dict.items():
        if len(event_times) != 0:
            sorted_adj[sorting_map[u], sorting_map[v]] = 1

    #Plot adjacency matrix in toned-down black and white
    plt.spy(sorted_adj, marker='.', markersize=0.1, precision=0)
    cumulative_community_size = 0
    for com_size in community_size:
        cumulative_community_size += com_size
        plt.axhline(cumulative_community_size, color='black', linewidth=1)
        plt.axvline(cumulative_community_size, color='black', linewidth=1)

    # plt.xticks(rotation=45)
    ticks = np.arange(0, num_nodes, 5000)
    plt.yticks(ticks, [f'{int(t / 1000)}{"K" if t >= 1000 else ""}' for t in ticks], fontsize=13)
    plt.xticks(ticks, [f'{int(t / 1000)}{"K" if t >= 1000 else ""}' for t in ticks], fontsize=13)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{result_file_path}/plots/community-structure-k-{num_classes}.png", format='png', dpi=200)
    # plt.savefig(f"{result_file_path}/plots/community-structure-k-{num_classes}.pdf", format='pdf')
