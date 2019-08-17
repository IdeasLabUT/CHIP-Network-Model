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
import parameter_estimation as estimate_utils
from spectral_clustering import spectral_cluster


result_file_path = f'{dataset_utils.get_script_path()}/storage/results/fb_chip_fit'

fit_chip = False
load_fb = True
plot_hawkes_params = True
plot_node_membership = True
plot_num_events = True
verbose = False
num_classes = 10

# load Facebook Wall-posts
if fit_chip or load_fb:
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
    node_membership = spectral_cluster(agg_adj, num_classes=10, verbose=False, plot_eigenvalues=True)

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

    cnt = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
                                                                            bp_mu[b_i, b_j],
                                                                            bp_alpha_beta_ratio[b_i, b_j],
                                                                            fb_duration, bp_size)
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
            num_events_block_pair[i, j] = len(block_pair_events[i][j])

    fig, ax = plt.subplots()
    labels = np.arange(1, num_classes + 1)
    im, _ = heatmap(num_events_block_pair, labels, labels, ax=ax, cmap="Greys", cbarlabel=" ")

    fig.tight_layout()
    # plt.show()
    plt.savefig(f"{result_file_path}/plots/num_block_pair_events.pdf")

    # plot block pair average number of events per node pair
    blocks, counts = np.unique(node_membership, return_counts=True)
    mean_num_events_block_pair = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            bp_size = counts[i] * counts[j] if i != j else counts[i] * (counts[i] - 1)
            mean_num_events_block_pair[i, j] = len(block_pair_events[i][j]) / bp_size

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

        fig.tight_layout()
        plt.savefig(f"{result_file_path}/plots/{param}-k-{num_classes}.pdf")
        # plt.show()

    # plot m
    fig, ax = plt.subplots()
    labels = np.arange(1, num_classes + 1)
    im, _ = heatmap(hawkes_params['alpha'] / hawkes_params['beta'], labels, labels, ax=ax, cmap="Greys",
                    cbarlabel=r"$m$")

    fig.tight_layout()
    # plt.show()
    plt.savefig(f"{result_file_path}/plots/m-k-{num_classes}.pdf")
