import numpy as np
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
        node_membership = np.argmax(node_membership, axis=1)

    return node_membership, community_membership


def simulate_univariate_hawkes(mu, alpha, beta, run_time, seed):
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
    Generate random numbers for hawkes params for generative models.

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