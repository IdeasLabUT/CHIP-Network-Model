import numpy as np
import generative_model_utils as utils


def community_generative_model(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta, end_time, seed=None):
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

    hawkes_seed = seed
    for c_i in range(num_classes):
        if len(community_membership[c_i]) == 0:
            continue

        for c_j in range(num_classes):
            if len(community_membership[c_j]) == 0:
                continue

            for b_i in community_membership[c_i]:
                for b_j in community_membership[c_j]:
                    # self events are not allowed
                    if b_i == b_j:
                        continue

                    # Seed has to change in order to get different event times for each node pair.
                    hawkes_seed = None if seed is None else hawkes_seed + 1

                    event_times = utils.simulate_univariate_hawkes(bp_mu[c_i, c_j],
                                                                   bp_alpha[c_i, c_j],
                                                                   bp_beta[c_i, c_j],
                                                                   end_time, seed=hawkes_seed)
                    event_dicts[(b_i, b_j)] = event_times

    return node_membership, event_dicts


if __name__ == "__main__":
    seed = 1
    number_of_nodes = 10
    class_probabilities = [0.2, 0.4, 0.1, 0.2, 0.1]
    num_of_classes = len(class_probabilities)
    end_time = 10
    bp_mu, bp_alpha, bp_beta = utils.generate_random_hawkes_params(num_of_classes,
                                                                   mu_range=(0.3, 1),
                                                                   alpha_range=(0.4, 0.9),
                                                                   beta_range=(0.95, 2),
                                                                   seed=seed)

    node_membership, event_dicts = community_generative_model(number_of_nodes,
                                                              class_probabilities,
                                                              bp_mu, bp_alpha, bp_beta,
                                                              end_time, seed=seed)

    print(node_membership, event_dicts.keys())
