# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""

import os
import pickle
import numpy as np
import dataset_utils
import matplotlib.pyplot as plt
import chip_generative_model as chip
from tick.hawkes import SimuHawkesExpKernels


def assign_class_membership(num_nodes, class_prob, one_hot=True):
    """
    Randomly assigns num_node nodes to one of the classes based on class_prob.

    :param num_nodes: (int) Total number of nodes
    :param class_prob: (list) Probability of class memberships from class 0 to K - 1
    :param one_hot: (bool) if True, node memberships will be returned with one hot encoding, otherwise a class number
                        will be returned

    :return: `node_membership`: membership of every node to one of K classes. num_nodes x num_classes (if one_hot)
        `community_membership`: list of nodes belonging to each class. num_classes x (varying size list)
    """
    num_classes = len(class_prob)

    node_membership = np.random.multinomial(1, class_prob, size=num_nodes)

    community_membership = node_membership_to_community_membership(node_membership, num_classes, is_one_hot=True)

    if not one_hot:
        node_membership = one_hot_to_class_assignment(node_membership)

    return node_membership, community_membership


def node_membership_to_community_membership(node_membership, n_classes, is_one_hot=False):
    """
    :param node_membership: membership of every node to one of K classes. num_nodes x num_classes (if one_hot)
                            num_nodes x 1 otherwise
    :param n_classes: Number of classes
    :param is_one_hot: True, if node_membership of one_hot encoded

    :return: list of nodes belonging to each class. num_classes x (varying size list)
    """
    community_membership = []

    if is_one_hot:
        for i in range(n_classes):
            community_membership.append(np.where(node_membership[:, i] == 1)[0])

        return community_membership

    for i in range(n_classes):
        community_membership.append(np.where(node_membership == i)[0])

    return community_membership


def simulate_univariate_hawkes(mu, alpha, beta, run_time, seed=None):
    """
    Simulates a univariate Hawkes based on the parameters.

    :param mu, alpha, beta: parameters of the Hawkes process
    :param run_time: End time of the simulation
    :param seed: (optional) Seed for the random process.
    :return: Hawkes event times
    """
    # this is due to tick's implementation of Hawkes process
    alpha = alpha / beta

    # Hawkes simulation
    n_nodes = 1  # dimension of the Hawkes process
    adjacency = alpha * np.ones((n_nodes, n_nodes))
    decays = beta * np.ones((n_nodes, n_nodes))
    baseline = mu * np.ones(n_nodes)
    hawkes_sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False, seed=seed)

    hawkes_sim.end_time = run_time
    hawkes_sim.simulate()
    event_times = hawkes_sim.timestamps[0]

    return event_times


def generate_random_hawkes_params(num_classes, mu_range, alpha_range, beta_range, seed=None):
    """
    Generate random numbers for Hawkes params for generative models.

    :param num_classes: number of class communities
    :param mu_range: (tuple) (min, max) range of mu.
    :param alpha_range: (tuple) (min, max) range of alpha.
    :param beta_range: (tuple) (min, max) range of beta.
    :param seed: seed of random generation

    :return: mu, alpha, beta
    """
    np.random.seed(seed)

    bp_mu = np.random.uniform(low=mu_range[0], high=mu_range[1], size=(num_classes, num_classes))
    bp_alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=(num_classes, num_classes))
    bp_beta = np.random.uniform(low=beta_range[0], high=beta_range[1], size=(num_classes, num_classes))

    return bp_mu, bp_alpha, bp_beta


def generate_theta_params_for_degree_corrected_community(num_nodes, dist, norm_sum_to):
    """
    Generate theta for every node to be used for the degree corrected Community Hawkes model.

    :param num_nodes: (int) Number of nodes in the network
    :param dist: (string) distribution to be drawn from. Either "exp" or "dirichlet"
    :param norm_sum_to: (string) Normalize all theta to sum to either "1" or "n" (number of nodes)

    :return: (list) theta values for each node
    """

    if dist != "exp" and dist != "dirichlet":
        exit("dist param can only be either exp or dirichlet")

    if norm_sum_to != "1" and norm_sum_to != "n":
        exit("norm_sum_to must be either '1' or 'n'")

    if dist == "dirichlet":
        theta = np.random.dirichlet(np.ones(num_nodes), 1)[0]
        return theta if norm_sum_to == "1" else theta * num_nodes

    theta = np.random.exponential(1, num_nodes)
    theta_normed_to_one = theta / sum(theta)

    return theta_normed_to_one if norm_sum_to == "1" else theta_normed_to_one * num_nodes


def event_dict_to_adjacency(num_nodes, event_dicts, dtype=np.float):
    """
    Converts event dict to unweighted adjacency matrix

    :param num_nodes: (int) Total number of nodes
    :param event_dicts: Edge dictionary of events between all node pair. Output of the generative models.
    :param dtype: data type of the adjacency matrix. Float is needed for the spectral clustering algorithm.

    :return: np array (num_nodes x num_nodes) Adjacency matrix with 1 between nodes where there is at least one event.
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        if len(event_times) != 0:
            adjacency_matrix[u, v] = 1

    return adjacency_matrix


def event_dict_to_aggregated_adjacency(num_nodes, event_dicts, dtype=np.float):
    """
    Converts event dict to weighted/aggregated adjacency matrix

    :param num_nodes: (int) Total number of nodes
    :param event_dicts: Edge dictionary of events between all node pair. Output of the generative models.
    :param dtype: data type of the adjacency matrix. Float is needed for the spectral clustering algorithm.

    :return: np array (num_nodes x num_nodes) Adjacency matrix where element ij denotes the number of events between
                                              nodes i an j.
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        adjacency_matrix[u, v] = len(event_times)

    return adjacency_matrix


def event_dict_to_block_pair_events(event_dicts, class_assignment, n_classes, is_for_tick=False):
    """
    Converts event_dicts to list of event lists for each block pair.

    :param event_dicts: Edge dictionary of events between all node pair. Output of the generative models.
    :param class_assignment: membership of every node to one of K classes. num_nodes x 1 (class of node i)
    :param n_classes: (int) total number of classes
    :param is_for_tick: (bool) if True, every list of event is wrapped in a list of length 1 to accommodate tick kernel
                        estimation.
    :return: (list) n_classes x n_classes where entry ij is a list of event lists between nodes in block i to nodes in
                    block j.
    """

    # Init block_pair_events
    block_pair_events = np.zeros((n_classes, n_classes), dtype=np.int).tolist()
    for i in range(n_classes):
        for j in range(n_classes):
            block_pair_events[i][j] = []

    for u, v in event_dicts:
        if is_for_tick and len(event_dicts[(u, v)]) == 0:
            continue

        if is_for_tick:
            block_pair_events[class_assignment[u]][class_assignment[v]].append([event_dicts[(u, v)]])
        else:
            block_pair_events[class_assignment[u]][class_assignment[v]].append(np.array(event_dicts[(u, v)]))

    return block_pair_events


def num_events_in_event_dict(event_dict):
    """
    Given an event dict, returns number of events

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.
    :return: (int) number of events
    """
    num_events = 0
    for _, event_times in event_dict.items():
        num_events += len(event_times)

    return num_events


def event_dict_to_event_list(event_dict):
    """
    Converts an event_dict to a list of events [from, to, timestamp] ordered by timestamp

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.

    :return: np.array num_events x 3 float
    """
    num_events = num_events_in_event_dict(event_dict)
    event_list = np.zeros((num_events, 3), np.float)

    i = 0
    for (u, v), event_times in event_dict.items():
        for t in event_times:
            event_list[i, :] = [u, v, t]
            i += 1

    # sort by timestamp
    event_list = event_list[event_list[:, 2].argsort()]

    return event_list


def one_hot_to_class_assignment(node_membership):
    """
    converts one-hot encoding of node_membership to class assignment

    :param node_membership: One-hot encoding of node_membership

    :return: 1-D np array with class of each node.
    """
    return np.argmax(node_membership, axis=1)


def calc_block_pair_size(class_assignment, n_classes):
    """
    Calculates the size of each block pair based on the class assignment.

    :param class_assignment: membership of every node to one of K classes. (1 x num_nodes)
    :param n_classes:  (int) total number of classes

    :return: K x K matrix, ij denotes the size of the block pair (b_i, b_j)
    """

    classes, class_size = np.unique(class_assignment, return_counts=True)
    if len(classes) != n_classes:
        exit("Fix calc_block_pair_size")

    # Sort classes sizes based on block index 0 to n_classes -1
    class_size = class_size[np.argsort(classes)]

    bp_size = np.ones((n_classes, n_classes)) * class_size
    # computing block size by |b_i| * |b_j|
    bp_size = bp_size * bp_size.T
    # Subtracting |b_i| from diagonals to get |b_i| * (|b_i| - 1) for diagonal block size
    bp_size = bp_size - np.diag(class_size)

    return bp_size


def scale_parameteres_by_block_pair_size(param, num_nodes, class_prob):
    """
    Calculates comparable hawkes parameters values based on the parameters for Block Hawkes model, for the Community
        Hawkes model, by dividing the parameter for each block pair by the expected size of that block pair.

    :param param: K x K matrix, ij denotes the hawkes parameter of the Block Hawkes process for block pair (b_i, b_j)
    :param num_nodes: (int) Total number of nodes
    :param class_prob: (list) Probability of class memberships from class 0 to K - 1

    :return: K x K matrix, ij denotes the mu of CHIP model for block pair (b_i, b_j)
    """
    num_classes = len(class_prob)

    expected_class_size = np.array(class_prob) * num_nodes
    expected_class_size_expanded = np.ones((num_classes, num_classes)) * expected_class_size

    # computing expected block size by |b_i| * |b_j|
    expected_block_size = expected_class_size_expanded * expected_class_size_expanded.T

    # Subtracting |b_i| from diagonals to get |b_i| * (|b_i| - 1) for diagonal block size
    expected_block_size = expected_block_size - np.diag(expected_class_size)

    community_model_mu = param / expected_block_size

    return community_model_mu


def plot_degree_count_histogram(aggregated_adjacency):
    """
    Plots a histogram of the weighted adjacency matrix.

    :param aggregated_adjacency: np array (num_nodes x num_nodes) Adjacency matrix where element ij denotes the
                                 number of events between nodes i an j.
    """
    n = aggregated_adjacency.shape[0]
    deg_count_flattened = np.reshape(aggregated_adjacency, (n * n))

    plt.hist(deg_count_flattened, bins=30)

    plt.xlabel('Event Count')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of the Count Matrix \n Mean Count: {np.mean(deg_count_flattened)}')
    plt.show()


def asymptotic_mean(mu, alpha, beta, run_time):
    """
    Calculates Hawkes asymptotic mean.
    """
    return (mu * run_time) / (1 - alpha / beta)


def asymptotic_var(mu, alpha, beta, run_time):
    """
    Calculates Hawkes asymptotic variance.
    """
    return (mu * run_time) / (1 - alpha / beta) ** 3


def simulate_community_hawkes(params=None, network_name=None, load_if_exists=False, verbose=False):
    """
    Simulates or loads a CHIP model based on the passed parameters.

    :param params: (dict) optional. Check variable `default_params` in the function for a list of parameters to change
    :param network_name: optional. name of the network to save or load.
    :param load_if_exists: If true, a network with `network_name` will be loaded if it exists.
    :param verbose: if True, details will be printed to the console.

    :return: event_dict, node_membership
    """
    generated_network_path = f'{dataset_utils.get_script_path()}/storage/results/generated_networks/'

    default_params = {'seed': None,
                      'number_of_nodes': 128,
                      'class_probabilities': [0.25, 0.25, 0.25, 0.25],
                      'end_time': 50,
                      'alpha': 7500,
                      'beta': 8000,
                      'mu_off_diag': 0.6,
                      'mu_diag': 1.8,
                      'num_nodes_to_scale': 128,
                      'alpha_diag': None,
                      'beta_diag': None,
                      'scale': True,
                      'n_cores': 1}

    # Load the network if existed
    if load_if_exists and network_name is not None:
        if os.path.isfile(generated_network_path + network_name + ".pckl"):
            with open(generated_network_path + network_name + ".pckl", 'rb') as handle:
                [event_dict, node_membership, params] = pickle.load(handle)

                if verbose:
                    print(params)
            return event_dict, node_membership

    if params is not None:
        default_params.update(params)

    seed = default_params['seed']
    number_of_nodes = default_params['number_of_nodes']
    class_probabilities = default_params['class_probabilities']
    num_of_classes = len(class_probabilities)
    end_time = default_params['end_time']
    burnin = None

    bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * default_params['alpha']
    bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * default_params['beta']
    bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * default_params['mu_off_diag']
    np.fill_diagonal(bp_mu, default_params['mu_diag'])

    if default_params['alpha_diag'] is not None:
        np.fill_diagonal(bp_alpha, default_params['alpha_diag'])

    if default_params['beta_diag'] is not None:
        np.fill_diagonal(bp_beta, default_params['beta_diag'])

    if default_params['scale']:
        n_scale = default_params['num_nodes_to_scale']
        bp_mu = scale_parameteres_by_block_pair_size(bp_mu, n_scale, class_probabilities)
        bp_alpha = scale_parameteres_by_block_pair_size(bp_alpha, n_scale, class_probabilities)
        bp_beta = scale_parameteres_by_block_pair_size(bp_beta, n_scale, class_probabilities)

    node_membership, event_dict = chip.community_generative_model(number_of_nodes,
                                                                  class_probabilities,
                                                                  bp_mu, bp_alpha, bp_beta,
                                                                  end_time, burnin=burnin,
                                                                  n_cores=default_params['n_cores'], seed=seed)

    node_membership = one_hot_to_class_assignment(node_membership)

    if network_name is not None:
        with open(generated_network_path + network_name + ".pckl", 'wb') as handle:
            pickle.dump([event_dict, node_membership, default_params], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return event_dict, node_membership
