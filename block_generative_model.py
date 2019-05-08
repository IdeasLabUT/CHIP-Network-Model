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

    hawkes_seed = 0
    for c_i in range(num_classes):
        # No member in the communit
        if len(community_membership[c_i]) == 0:
            continue

        for c_j in range(num_classes):
            # No member in the community
            if len(community_membership[c_j]) == 0:
                continue

            # generating within community events, when the community only has one member
            if c_i == c_j and len(community_membership[c_j]) == 1:
                continue

            # In case two block pairs have same Hawkes params, we still need different generated event times
            hawkes_seed = None if seed is None else hawkes_seed + 1

            event_times = utils.simulate_univariate_hawkes(bp_mu[c_i, c_j],
                                                           bp_alpha[c_i, c_j],
                                                           bp_beta[c_i, c_j],
                                                           end_time, seed=seed)
            num_events = len(event_times)

            # self events are not allowed. Nodes must be sampled without replacement for within community events.
            if c_i != c_j:
                block_i_nodes = np.random.choice(community_membership[c_i], num_events, replace=True)
                block_j_nodes = np.random.choice(community_membership[c_j], num_events, replace=True)
            else:
                block_i_nodes = np.empty(num_events, dtype=int)
                block_j_nodes = np.empty(num_events, dtype=int)

                for bn in range(num_events):
                    block_i_nodes[bn], block_j_nodes[bn] = np.random.choice(community_membership[c_i], 2, replace=False)

            for e in range(num_events):
                node_pair = (block_i_nodes[e], block_j_nodes[e])
                if node_pair not in event_dicts:
                    event_dicts[node_pair] = []

                event_dicts[node_pair].append(event_times[e])

    return node_membership, event_dicts


if __name__ == "__main__":
    seed = 1
    number_of_nodes = 8
    class_probabilities = [0.2, 0.4, 0.1, 0.2, 0.1]
    num_of_classes = len(class_probabilities)
    end_time = 10
    bp_mu, bp_alpha, bp_beta = utils.generate_random_hawkes_params(num_of_classes,
                                                                   mu_range=(0.1, 0.3),
                                                                   alpha_range=(0.2, 0.4),
                                                                   beta_range=(0.5, 1),
                                                                   seed=seed)

    node_membership, event_dicts = block_generative_model(number_of_nodes,
                                                          class_probabilities,
                                                          bp_mu, bp_alpha, bp_beta,
                                                          end_time, seed=seed)

    print(node_membership, event_dicts.keys())
    print(utils.event_dict_to_adjacency(number_of_nodes, event_dicts))
    print(utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts))

