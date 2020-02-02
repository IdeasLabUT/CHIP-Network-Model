# -*- coding: utf-8 -*-
"""
"Exploratory Analysis of the Enron Dataset using CHIP"

Here we fit CHIP to the Enron dataset and analyze various aspects of the fitted model.

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


result_file_path = '/shared/Results/CommunityHawkes/pickles/enron_chip_fit'

fit_chip = False
load_dataset = False
plot_hawkes_params = False
plot_node_membership = False
plot_num_events = False
analyze_position = True
verbose = False
num_classes = 2


# Load combined Enron Dataset
if fit_chip or load_dataset or plot_num_events:
    tic = time.time()
    *_, enron_combined_tuple, _ = dataset_utils.load_enron_train_test(remove_nodes_not_in_train=False)
    data_event_dict, data_num_nodes, data_duration = enron_combined_tuple

    toc = time.time()

    print(f"Loaded the dataset in {toc - tic:.1f}s")

    num_events = utils.num_events_in_event_dict(data_event_dict)
    if verbose:
        print("Num Nodes:", data_num_nodes, "Duration:", data_duration,
              "Num Edges:", num_events)

# fit dataset
if fit_chip:
    tic = time.time()
    agg_adj = utils.event_dict_to_aggregated_adjacency(data_num_nodes, data_event_dict)
    adj = utils.event_dict_to_adjacency(data_num_nodes, data_event_dict)
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
        print("Community assignment prob:", np.unique(node_membership, return_counts=True)[1] / data_num_nodes)

    tic = time.time()
    bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership, data_duration,
                                                                            1e-10 / data_duration)
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
    block_pair_events = utils.event_dict_to_block_pair_events(data_event_dict, node_membership, num_classes)

    cnt = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
                                                                            bp_mu[b_i, b_j],
                                                                            bp_alpha_beta_ratio[b_i, b_j],
                                                                            data_duration, bp_size)
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
        pickle.dump([data_num_nodes, num_events, data_duration,
                     node_membership,
                     hawkes_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{result_file_path}/all_model_params-k-{num_classes}.pckl', 'rb') as handle:
        [data_num_node, num_events, data_duration,
         node_membership,
         hawkes_params] = pickle.load(handle)


print("Diag mu mean: ", np.mean(hawkes_params['mu'][np.eye(num_classes, dtype=bool)]))
print("off-diag mu mean: ", np.mean(hawkes_params['mu'][~np.eye(num_classes, dtype=bool)]))

if plot_num_events:
    block_pair_events = utils.event_dict_to_block_pair_events(data_event_dict, node_membership, num_classes)

    # plot number of events per block pair
    num_events_block_pair = np.zeros((num_classes, num_classes), dtype=np.int)
    for i in range(num_classes):
        for j in range(num_classes):
            num_events_block_pair[i, j] = len(np.concatenate(block_pair_events[i][j]))

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
            mean_num_events_block_pair[i, j] = len(np.concatenate(block_pair_events[i][j])) / bp_size

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
    ind = np.arange(1, num_classes + 1)    # the x locations for the groups
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
    # ax.set_ylim(0, 25)
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
    plt.savefig(f"{result_file_path}/plots/m-k-{num_classes}.pdf")
    # plt.show()

    # plot mu / (1 - m)
    fig, ax = plt.subplots()
    labels = np.arange(1, num_classes + 1)
    m = hawkes_params['alpha'] / hawkes_params['beta']
    expected = hawkes_params['mu'] / (1 - m)
    im, _ = heatmap(expected, labels, labels, ax=ax, cmap="Greys",
                    cbarlabel=r"$\mu/(1-m)$")

    fig.tight_layout()
    # plt.show()
    plt.savefig(f"{result_file_path}/plots/mu-over-1-m-k-{num_classes}.pdf")

    # print(f"Diag mean:  {np.mean(expected[np.eye(num_classes, dtype=bool)]):.3e}", end=', ')
    # print(f"sd: {np.std(expected[np.eye(num_classes, dtype=bool)]):.3e}")
    # print(f"off-diag mean: {np.mean(expected[~np.eye(num_classes, dtype=bool)]):.3e}", end=', ')
    # print(f"sd: {np.std(expected[~np.eye(num_classes, dtype=bool)]):.3e}")

if analyze_position:
    # load dict of node id to index map
    dataset_path = '/shared/DataSets/EnronPriebe2009'
    with open(f'{dataset_path}/node_id_to_position_dict_map.pckl', 'rb') as handle:
        node_to_idx = pickle.load(handle)

    # get the position of each node
    idx_to_position = {}
    with open(f'{dataset_path}/position.csv', 'r') as f:
        for i, l in enumerate(f):
            if i + 1 in node_to_idx:
                idx_to_position[node_to_idx[i + 1]] = l.strip()

    # Explore the memberships
    for c_idx in range(num_classes):
        print("Class: ", c_idx + 1)
        print("Positions:")

        na_cnt = 0
        emp_cnt = 0
        non_emp_cnt = 0
        for i, c in enumerate(node_membership):
            if c != c_idx:
                continue

            p = idx_to_position[i]
            if p == 'NA':
                na_cnt += 1
            elif p == 'Employee':
                emp_cnt += 1
            else:
                non_emp_cnt += 1

            print(p, end=', ')

        tot_nodes = np.sum(node_membership == c_idx)
        print('\n')
        print(f'Counts -> Employees: {emp_cnt} - Higher Ups: {non_emp_cnt} - NA\'s: {na_cnt}')
        print(f'Percentage -> Employees: {100 * emp_cnt / tot_nodes:.1f}% - '
              f'Higher Ups: {100 * non_emp_cnt / tot_nodes:.1f}% - NA\'s: {100 * na_cnt / tot_nodes:.1f}%')
        print(f'Percentage excluding NA\'s -> Employees: {100 * emp_cnt / (tot_nodes - na_cnt):.1f}% - '
              f'Higher Ups: {100 * non_emp_cnt / (tot_nodes - na_cnt):.1f}%')
        print()
