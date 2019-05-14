import numpy as np
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkesExpKernels


def assign_class_membership(num_nodes, class_prob, one_hot=True):
    """
    Randomly assigns num_node nodes to one of the classes based on class_prob.

    :rtype: `node_membership`: membership of every node to one of K classes. num_nodes x num_classes (if one_hot)
            `community_membership`: list of nodes belonging to each class. num_classes x (varying size list)
    :param num_nodes: (int) Total number of nodes
    :param class_prob: (list) Probability of class memberships from class 0 to K - 1
    :param one_hot: (bool) if True, node memberships will be returned with one hot encoding, otherwise a class number
                        will be returned
    """
    num_classes = len(class_prob)

    node_membership = np.random.multinomial(1, class_prob, size=num_nodes)
    community_membership = []

    for i in range(num_classes):
        community_membership.append(np.where(node_membership[:, i] == 1)[0])

    if not one_hot:
        node_membership = one_hot_to_class_assignment(node_membership)

    return node_membership, community_membership


def simulate_univariate_hawkes(mu, alpha, beta, run_time, seed=None):
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
    """
    np.random.seed(seed)

    bp_mu = np.random.uniform(low=mu_range[0], high=mu_range[1], size=(num_classes, num_classes))
    bp_alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=(num_classes, num_classes))
    bp_beta = np.random.uniform(low=beta_range[0], high=beta_range[1], size=(num_classes, num_classes))

    return bp_mu, bp_alpha, bp_beta


def event_dict_to_adjacency(num_nodes, event_dicts, dtype=np.float):
    """

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


def one_hot_to_class_assignment(node_membership):
    """
    converts one-hot encoding of node_membership to class assignment

    :param node_membership: One-hot encoding of node_membership
    :return: 1-D np array with class of each node.
    """
    return np.argmax(node_membership, axis=1)


def scale_parameteres_by_block_pair_size(param, num_nodes, class_prob):
    """
    Calculates comparable hawkes parameters values based on the parameters for Block Hawkes model, for the Community
        Hawkes model, by dividing the parameter for each block pair by the expected size of that block pair.
    :param param: K x K matrix, ij denotes the hawkes parameter of the Block Hawkes process for block pair (b_i, b_j)
    :param num_nodes: (int) Total number of nodes
    :param class_prob: (list) Probability of class memberships from class 0 to K - 1
    :return: K x K matrix, ij denotes the mu of the Community Hawkes process for block pair (b_i, b_j)
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