import numpy as np
import block_local_search as bls
from scipy.optimize import minimize
from scipy.stats import multinomial
import generative_model_utils as utils
from spectral_clustering import spectral_cluster, regularized_spectral_cluster


def fit_block_model(event_dict, num_nodes, duration, num_classes, local_search_max_iter, local_search_n_cores,
                    verbose=False):
    # agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)
    adj = utils.event_dict_to_adjacency(num_nodes, event_dict)

    # Running spectral clustering
    node_membership = regularized_spectral_cluster(adj, num_classes=num_classes)

    if local_search_max_iter > 0 and num_classes > 1:
        node_membership, bp_mu, bp_alpha, bp_beta = bls.block_local_search(event_dict, num_classes, node_membership,
                                                                           duration,
                                                                           local_search_max_iter, local_search_n_cores,
                                                                           return_fitted_param=True, verbose=False)
        bp_events = event_dict_to_combined_block_pair_events(event_dict, node_membership, num_classes)

    else:
        bp_events = event_dict_to_combined_block_pair_events(event_dict, node_membership, num_classes)

        bp_mu, bp_alpha, bp_beta = estimate_hawkes_params(bp_events, node_membership, duration, num_classes)

    # Printing information about the fit
    if verbose:
        _, block_count = np.unique(node_membership, return_counts=True)
        class_prob = block_count / sum(block_count)

        print(f"Membership percentage: ", class_prob)

        print("Mu:")
        print(bp_mu)

        print("\nAlpha:")
        print(bp_alpha)

        print("\nBeta:")
        print(bp_beta)

    return node_membership, bp_mu, bp_alpha, bp_beta, bp_events


def estimate_hawkes_param_and_calc_log_likelihood(event_dict, node_membership, duration, num_classes,
                                                  add_com_assig_log_prob=False):

    bp_events = event_dict_to_combined_block_pair_events(event_dict, node_membership, num_classes)

    bp_mu, bp_alpha, bp_beta = estimate_hawkes_params(bp_events, node_membership, duration, num_classes)

    log_likelihood = calc_full_log_likelihood(bp_events, node_membership,
                                              bp_mu, bp_alpha, bp_beta,
                                              duration, num_classes, add_com_assig_log_prob)

    return log_likelihood, (bp_mu, bp_alpha, bp_beta)


def estimate_hawkes_params(bp_events, node_membership, duration, num_classes):
    bp_mu = np.zeros((num_classes, num_classes), dtype=np.float)
    bp_alpha = np.zeros((num_classes, num_classes), dtype=np.float)
    bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)

    # block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, num_classes)

    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            bp_mu[b_i, b_j], bp_alpha[b_i, b_j], bp_beta[b_i, b_j] = estimate_all_bp_from_events(bp_events[b_i][b_j],
                                                                                                 duration, bp_size,
                                                                                                 (1e-2, 2e-2, 2e-5))

    return bp_mu, bp_alpha, bp_beta


def calc_full_log_likelihood(bp_events, node_membership, mu, alpha, beta, duration, num_classes,
                             add_com_assig_log_prob=False):
    ll = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            ll += block_pair_conditional_log_likelihood(bp_events[b_i][b_j],
                                                        mu[b_i, b_j], alpha[b_i, b_j], beta[b_i, b_j],
                                                        duration, bp_size)

    if add_com_assig_log_prob:
        # Adding the log probability of the community assignments to the full log likelihood
        n_nodes = len(node_membership)
        _, block_count = np.unique(node_membership, return_counts=True)
        class_prob_mle = block_count / sum(block_count)
        rv_multi = multinomial(n_nodes, class_prob_mle)
        log_prob_community_assignment = rv_multi.logpmf(block_count)

        ll += log_prob_community_assignment

    return ll


def event_dict_to_combined_block_pair_events(event_dict, class_assignment, n_classes):
    """
    BLOCK MODEL'S BLOCK PAIR EVENTS SHOULD NOT BE MISTAKEN FOR CHP MODEL'S BLOCK PAIR EVENTS! THEY ARE STRUCTURALLY
    DIFFERENT, ALTHOUGH THEY BOTH CONTAIN THE SIMILAR INFORMATION.

    Converts event_dicts to list of event lists for each block pair.

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.
    :param class_assignment: membership of every node to one of K classes. num_nodes x 1 (class of node i)
    :param n_classes: (int) total number of classes
    :return: (list) n_classes x n_classes where entry ij is a sorted np array of events between nodes in block i to nodes in
                    block j.
    """

    # Init block_pair_events
    block_pair_events = np.zeros((n_classes, n_classes), dtype=np.int).tolist()
    for i in range(n_classes):
        for j in range(n_classes):
            block_pair_events[i][j] = []

    for u, v in event_dict:
        block_pair_events[class_assignment[u]][class_assignment[v]].extend(event_dict[(u, v)])

    for i in range(n_classes):
        for j in range(n_classes):
            block_pair_events[i][j] = np.sort(block_pair_events[i][j])

    return block_pair_events


def compute_wijs_recursive(bp_events, beta):
    n_events = len(bp_events)
    if n_events < 1:
        return 0

    wijs = np.zeros(n_events)
    for i in range(1, n_events):
        wijs[i] = np.exp(-beta * (bp_events[i] - bp_events[i - 1])) * (1 + wijs[i - 1])

    return wijs


def block_pair_conditional_log_likelihood(bp_events, mu, alpha, beta, end_time, block_pair_size):
    """

    :param block_pair_size: Size of the block pair. bp_events may not include an entry for node_pairs with no
                            interactions, in that case, we need to add (-mu * end_time) to the likelihood for each
                            missing node pair.
    """
    ll = 0
    bp_n_events = len(bp_events)

    # TODO: what to do when we have beta, alpha or mu == 0?
    if mu == 0:
        mu = 1e-10 / end_time

    if alpha == 0 or beta == 0:
        alpha = 0
        beta = 1  # Doesn't matter what beta is since alpha is set to 0.

    if bp_n_events > 0:
        # first sum
        ll += (alpha / beta) * np.sum(np.exp(-beta * (end_time - bp_events)) - 1)

        # second recursive sum
        ll += np.sum(np.log(mu + alpha * compute_wijs_recursive(bp_events, beta)))

    # third term
    ll -= mu * end_time

    # second part of the log-likelihood
    ll -= bp_n_events * np.log(block_pair_size)

    return ll


def neg_log_likelihood_all_bp(param, bp_events, end_time, block_pair_size):
    alpha = param[0]
    beta = param[1]
    mu = param[2]
    return -block_pair_conditional_log_likelihood(bp_events, mu, alpha, beta,
                                                  end_time, block_pair_size)


def estimate_all_bp_from_events(bp_events, end_time, block_pair_size, init_param=(1e-2,2e-2,2e-5), return_detail=False):
    res = minimize(neg_log_likelihood_all_bp, init_param, method='L-BFGS-B',
                   bounds=((0, None), (0, None), (0, None)), jac=None,
                   args=(bp_events, end_time, block_pair_size))

    if return_detail:
        return res.x, res

    alpha, beta, mu = res.x

    return mu, alpha, beta


# def block_pair_conditional_log_likelihood(bp_events, mu, alpha, beta, end_time, block_pair_size):
#     """
#
#     :param block_pair_size: Size of the block pair. bp_events may not include an entry for node_pairs with no
#                             interactions, in that case, we need to add (-mu * end_time) to the likelihood for each
#                             missing node pair.
#     """
#     ll = 0
#     bp_n_events = 0
#
#     for np_events in bp_events:
#         ll += -mu * end_time
#
#         if len(np_events) == 0:
#             continue
#
#         bp_n_events += len(np_events)
#
#         second_inner_sum = (alpha / beta) * np.sum(np.exp(-beta * (end_time - np_events)) - 1)
#         third_inner_sum = np.sum(np.log(mu + alpha * compute_wijs_recursive(np_events, beta)))
#
#         ll += second_inner_sum + third_inner_sum
#
#     num_missing_node_pairs = block_pair_size - len(bp_events)
#     ll += num_missing_node_pairs * -mu * end_time
#
#     # second part of the log-likelihood
#     ll -= bp_n_events * np.log(block_pair_size)
#
#     return ll
