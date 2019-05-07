import numpy as np
import generative_model_utils as utils


def block_generative_model(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta, end_time, seed=None):
    """

    :param num_nodes: (int) Total number of nodes
    :param class_prob: (list) Probability of class memberships from class 0 to K - 1
    :param bp_mu: K*2 x K*2 matrix where entry ij denotes the mu of Hawkes process for block pair (b_i, b_j)
    :param bp_alpha: K*2 x K*2 matrix where entry ij denotes the alpha of Hawkes process for block pair (b_i, b_j)
    :param bp_beta: K*2 x K*2 matrix where entry ij denotes the beta of Hawkes process for block pair (b_i, b_j)
    :param end_time: end_time of hawkes simulation
    :param seed: seed of all random processes
    """
    np.random.seed(seed)

    num_classes = len(class_prob)
    node_membership, community_membership = utils.assign_class_membership(num_nodes, class_prob)

    event_dicts = {}

    for i in range(num_classes):
        if len(community_membership[i]) == 0:
            continue

        for j in range(num_classes):
            if i == j or len(community_membership[j]) == 0:
                continue

            event_times = utils.simulate_univariate_hawkes(bp_mu[i, j], bp_alpha[i, j], bp_beta[i, j], end_time, seed=seed)
            num_events = len(event_times)

            block_i_nodes = np.random.choice(community_membership[i], num_events, replace=True)
            block_j_nodes = np.random.choice(community_membership[j], num_events, replace=True)

            for e in range(num_events):
                node_pair = (block_i_nodes[e], block_j_nodes[e])
                if node_pair not in event_dicts:
                    event_dicts[node_pair] = []

                event_dicts[node_pair].append(event_times[e])

    return node_membership, event_dicts


if __name__ == "__main__":
    seed = 1
    number_of_nodes = 5
    class_probabilities = [0.2, 0.4, 0.1, 0.3]
    num_of_classes = len(class_probabilities)
    end_time = 10
    bp_mu, bp_alpha, bp_beta = utils.generate_random_hawkes_params(num_of_classes, (0.3, 1), (0.4, 0.9), (0.95, 2), seed=seed)

    print(block_generative_model(number_of_nodes, class_probabilities, bp_mu, bp_alpha, bp_beta, end_time, seed=seed))


