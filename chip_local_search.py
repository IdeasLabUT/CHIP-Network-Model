# -*- coding: utf-8 -*-
"""
@author: Makan Arastuie
"""

import time
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import generative_model_utils as utils
import parameter_estimation as estimate_utils
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster
import model_fitting_utils as fit_utils


def calc_node_neigh_solutions(event_dict, n_classes, duration, node_membership, agg_adj, beta, log_lik_init, node_batch):
    """
    Calculates the log-likelihood of neighboring solutions of a batch of nodes by changing their membership. If a higher
    log-likelihood was achieved the best solution will be returned, else a tuple of three np.nan is returned.

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.
    :param n_classes: (int) total number of classes/blocks
    :param duration: (int) Duration of the network
    :param node_membership: (list) membership of every node to one of K classes
    :param agg_adj: aggregated/weighted adjacency of the network
    :param beta: K x K np array of block pairs beta. This is fixed for every solution to lower time complexity. Only mu
                 and m are estimated for each neighboring solution
    :param log_lik_init: (float) base log-likelihood
    :param node_batch: (list) nodes in the current batch

    :return: (node index, best class index, log_likelihood)
    """

    best_neigh = (np.nan, np.nan, np.nan)
    log_lik = log_lik_init
    # node_membership = node_membership.copy()

    for n_i in node_batch:
        n_i_class = node_membership[n_i]

        # Adding a constraint to maintain the number of blocks.
        if np.sum(node_membership == n_i_class) <= 2:
            continue

        for c_i in range(n_classes):
            if c_i == n_i_class:
                continue

            # update node_membership temporarily
            node_membership[n_i] = c_i

            # Eval the aprox log_lik of this neighbor, by est its mu and alpha/beta and using previous beta.
            neigh_mu, neigh_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                                          duration,
                                                                                          default_mu=1e-10 / duration)
            neigh_alpha = neigh_alpha_beta_ratio * beta

            block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, n_classes)
            neigh_log_lik = fit_utils.calc_full_log_likelihood(block_pair_events, node_membership,
                                                               neigh_mu, neigh_alpha, beta,
                                                               duration, n_classes, add_com_assig_log_prob=False)

            # if log_lik if this neighbor is better than the "so far" best neighbor, use this neighbors as the best.
            if log_lik < neigh_log_lik:
                log_lik = neigh_log_lik
                best_neigh = (n_i, c_i, log_lik)

            node_membership[n_i] = n_i_class

    return best_neigh


def chp_local_search(event_dict, n_classes, node_membership_init, duration, max_iter=100, n_cores=-1,
                     return_fitted_param=False, verbose=True):
    """
    Performs local search / hill climbing to increase log-likelihood of the model by switching the community of a single
    node at a time. For every neighboring solution only mu and m are estimated, beta is fixed to the base solution to
    lower time complexity.

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.
    :param n_classes: (int) total number of classes/blocks
    :param node_membership_init: (list) initial membership of every node to one of K classes. Usually output of the
                                 spectral clustering
    :param duration: (int) Duration of the network
    :param max_iter: (int) maximum number of iterations to be performed by local search.
    :param n_cores: (int) number of cores to be used to parallelize the search. If -1, use all available cores.
    :param return_fitted_param: if True, return the Hawkes parameters for the model as well.
    :param verbose: If True, prints more information on local search.

    :return: local optimum node_membership if `return_fitted_param` is false.
    """
    n_nodes = len(node_membership_init)
    nodes = np.arange(n_nodes)
    node_membership = node_membership_init
    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict, dtype=np.int)

    # estimate initial params of CHIP and its log-likelihood
    (mu,
     alpha,
     beta,
     alpha_beta_ratio) = fit_utils.estimate_bp_hawkes_params(event_dict, node_membership, duration, n_classes)

    block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, n_classes)
    init_log_lik = fit_utils.calc_full_log_likelihood(block_pair_events, node_membership, mu, alpha, beta,
                                                      duration, n_classes, add_com_assig_log_prob=False)

    log_lik = init_log_lik
    n_cores = n_cores if n_cores > 0 else multiprocessing.cpu_count()
    batch_size = np.int(n_nodes / n_cores) + 1

    for iter in range(max_iter):
        if verbose:
            print(f"Iteration {iter}...", end='\r')

        # for each of the (k-1)*n neighboring solutions
        possible_solutions = Parallel(n_jobs=n_cores)(delayed(calc_node_neigh_solutions)
                                                      (event_dict, n_classes, duration, node_membership, agg_adj,
                                                       beta, log_lik, nodes[batch_size * ii: batch_size * (ii + 1)])
                                                      for ii in range(n_cores))

        possible_solutions = np.array(possible_solutions)

        # if all returned log-likelihoods are np.nan, break. You're at a local optima.
        if np.all(np.isnan(possible_solutions[:, 2])):
            if verbose:
                print(f"Local solution found with {iter} iterations.")
            break

        max_ll_neigh_idx = np.nanargmax(possible_solutions[:, 2])

        # if a good neighbor was found, update all CHIP params, and go for the next iteration.
        node_membership[int(possible_solutions[max_ll_neigh_idx, 0])] = int(possible_solutions[max_ll_neigh_idx, 1])
        (mu,
         alpha,
         beta,
         alpha_beta_ratio) = fit_utils.estimate_bp_hawkes_params(event_dict, node_membership, duration, n_classes)

        block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, n_classes)
        log_lik = fit_utils.calc_full_log_likelihood(block_pair_events, node_membership, mu, alpha, beta,
                                                     duration, n_classes, add_com_assig_log_prob=False)

        if iter == max_iter - 1:
            print("Warning: Max iter reached!")

    if verbose:
        print(f"likelihood went from {init_log_lik:.4f} to {log_lik:.4f}. "
              f"{100 * np.abs((log_lik - init_log_lik) / init_log_lik):.2f}% increase.")

    if return_fitted_param:
        return node_membership, mu, alpha, beta

    return node_membership


def chp_local_search_single_core(event_dict, n_classes, node_membership_init, duration, max_iter=100, verbose=True):
    """
    This function is only here for speed comparisons against the multi-core version. All parameters are the same as
    `chip_local_search`.
    """
    n_nodes = len(node_membership_init)
    node_membership = node_membership_init
    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict, dtype=np.int)

    # estimate initial params of CHIP and its log-likelihood
    (mu,
     alpha,
     beta,
     alpha_beta_ratio) = fit_utils.estimate_bp_hawkes_params(event_dict, node_membership, duration, n_classes)

    block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, n_classes)
    init_log_lik = fit_utils.calc_full_log_likelihood(block_pair_events, node_membership, mu, alpha, beta,
                                                      duration, n_classes, add_com_assig_log_prob=False)

    log_lik = init_log_lik

    for iter in range(max_iter):
        if verbose:
            print(f"Iteration {iter}...", end='\r')

        # best neighbor will hold the best node_membership update in the form of (node_index, updated_class_membership)
        best_neigh = None

        # for each of the (k-1)*n neighboring solutions
        for n_i in range(n_nodes):
            n_i_class = node_membership[n_i]

            for c_i in range(n_classes):
                if c_i == n_i_class:
                    continue
                # update node_membership temporarily
                node_membership[n_i] = c_i

                # Eval the aprox log_lik of this neighbor, by est its mu and alpha/beta and using previous beta.
                neigh_mu, neigh_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                                              duration,
                                                                                              default_mu=1e-10 / duration)
                neigh_alpha = neigh_alpha_beta_ratio * beta

                block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, n_classes)
                neigh_log_lik = fit_utils.calc_full_log_likelihood(block_pair_events, node_membership,
                                                                   neigh_mu, neigh_alpha, beta,
                                                                   duration, n_classes, add_com_assig_log_prob=False)

                # if log_lik if this neighbor is better than the "so far" best neighbor, use this neighbors as the best.
                if log_lik < neigh_log_lik:
                    log_lik = neigh_log_lik
                    best_neigh = (n_i, c_i)

                node_membership[n_i] = n_i_class

        # if no neighbor seem to increase log_lik, break. You're at a local optima.
        if best_neigh is None:
            if verbose:
                print(f"Local solution found with {iter} iterations.")
            break

        # if a good neighbor was found, update all CHIP params, and go for the next iteration.
        node_membership[best_neigh[0]] = best_neigh[1]
        (mu,
         alpha,
         beta,
         alpha_beta_ratio) = fit_utils.estimate_bp_hawkes_params(event_dict, node_membership, duration, n_classes)

        block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, n_classes)
        log_lik = fit_utils.calc_full_log_likelihood(block_pair_events, node_membership, mu, alpha, beta,
                                                     duration, n_classes, add_com_assig_log_prob=False)

    if verbose:
        print(f"likelihood went from {init_log_lik:.4f} to {log_lik:.4f}. "
              f"{100 * np.abs((log_lik - init_log_lik) / init_log_lik):.2f}% increase.")

    return node_membership


# Examples of running local search on CHIP model.
if __name__ == '__main__':
    n_classes = 4
    n_nodes = 1024
    duration = 50

    params = {'number_of_nodes': n_nodes,
              'alpha': 0.6,
              'beta': 0.8,
              'mu_off_diag': 0.8,
              'mu_diag': 1.6,
              'end_time': duration,
              'class_probabilities': np.ones(n_classes) / n_classes,
              'n_cores': -1}

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params,
                                                                         network_name="local_seach_test_networks",
                                                                         load_if_exists=False)

    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
    spectral_node_membership = spectral_cluster(agg_adj, num_classes=n_classes)
    sc_rand = adjusted_rand_score(true_class_assignments, spectral_node_membership)
    print(f"SC Rand index: {sc_rand:.3f}")

    print("Parallel")
    tic = time.time()
    local_search_node_membership = chp_local_search(event_dict, n_classes, spectral_node_membership, duration,
                                                    max_iter=10, n_cores=34, verbose=True)
    toc = time.time()
    print(f"local search took {toc - tic:.2f}s.")

    sc_rand = adjusted_rand_score(true_class_assignments, local_search_node_membership)
    print(f"Local search Rand index: {sc_rand:.3f}")


    print("Single core")
    tic = time.time()
    local_search_node_membership = chp_local_search_single_core(event_dict, n_classes, spectral_node_membership,
                                                                duration, max_iter=10, verbose=True)
    toc = time.time()
    print(f"local search took {toc - tic:.2f}s.")
    sc_rand = adjusted_rand_score(true_class_assignments, local_search_node_membership)
    print(f"Local search Rand index: {sc_rand:.3f}")
