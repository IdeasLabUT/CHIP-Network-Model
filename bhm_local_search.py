# -*- coding: utf-8 -*-
"""
@author: Makan Arastuie
"""

import time
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import generative_model_utils as utils
import bhm_generative_model as bhm
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import regularized_spectral_cluster
import bhm_parameter_estimation as bhm_estimate_utils


def calc_node_neigh_solutions(event_dict, n_classes, duration, node_membership, log_lik_init, node_batch):
    """
    Calculates the log-likelihood of neighboring solutions of a batch of nodes by changing their membership. If a higher
    log-likelihood was achieved the best solution will be returned, else a tuple of three np.nan is returned.

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.
    :param n_classes: (int) total number of classes
    :param duration: (int) Duration of the network
    :param node_membership: (list) membership of every node to one of K classes
    :param log_lik_init: (float) base log-likelihood of the current solution
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

            # Eval the aprox log_lik of this neighbor, by est all block parameters.
            (neigh_log_lik,
             fitted_params) = bhm_estimate_utils.estimate_hawkes_param_and_calc_log_likelihood(event_dict,
                                                                                               node_membership,
                                                                                               duration, n_classes,
                                                                                               False)

            # if log_lik if this neighbor is better than the "so far" best neighbor, use this neighbors as the best.
            if log_lik < neigh_log_lik:
                log_lik = neigh_log_lik
                best_neigh = (n_i, c_i, log_lik)

            node_membership[n_i] = n_i_class

    return best_neigh


def block_local_search(event_dict, n_classes, node_membership_init, duration, max_iter=100, n_cores=-1,
                       return_fitted_param=False, verbose=False):
    """
    Performs local search / hill climbing to increase log-likelihood of the model by switching the community of a single
    node at a time.

    :param event_dict: Edge dictionary of events between all node pair. Output of the generative models.
    :param n_classes: (int) total number of classes
    :param node_membership_init: (list) initial membership of every node to one of K classes.
    :param duration: (int) Duration of the network.
    :param max_iter: (int) maximum number of iterations to be performed by local search.
    :param n_cores: (int) number of cores to be used to parallelize the search. If -1, use all available cores.
    :param return_fitted_param: if True, return the Hawkes parameters for the model as well.
    :param verbose: If True, prints more information on local search.

    :return: local optimum node_membership if `return_fitted_param` is false.
    """
    n_nodes = len(node_membership_init)
    nodes = np.arange(n_nodes)
    node_membership = node_membership_init

    # estimate initial params of block model and its log-likelihood
    (init_log_lik,
     fitted_params) = bhm_estimate_utils.estimate_hawkes_param_and_calc_log_likelihood(event_dict, node_membership,
                                                                                       duration, n_classes,
                                                                                       add_com_assig_log_prob=False)
    log_lik = init_log_lik
    n_cores = n_cores if n_cores > 0 else multiprocessing.cpu_count()
    batch_size = np.int(n_nodes / n_cores) + 1

    # print(n_cores)
    for iter in range(max_iter):
        if verbose:
            print(f"Iteration {iter}...", end='\r')

        tic = time.time()
        # for each of the (k-1)*n neighboring solutions
        possible_solutions = Parallel(n_jobs=n_cores)(delayed(calc_node_neigh_solutions)
                                                      (event_dict, n_classes, duration, node_membership, log_lik,
                                                       nodes[batch_size * ii: batch_size * (ii + 1)])
                                                      for ii in range(n_cores))
        toc = time.time()
        print(f"Iter {iter}, took: {(toc - tic)/3600:.3f}h")

        possible_solutions = np.array(possible_solutions)

        # if all returned log-likelihoods are np.nan, break. You're at a local optima.
        if np.all(np.isnan(possible_solutions[:, 2])):
            if verbose:
                print(f"Local solution found with {iter} iterations.")
            break

        max_ll_neigh_idx = np.nanargmax(possible_solutions[:, 2])
        best_node_to_switch = int(possible_solutions[max_ll_neigh_idx, 0])
        best_class_to_switch_to = int(possible_solutions[max_ll_neigh_idx, 1])

        # if a good neighbor was found, update best log_lik, and go for the next iteration.
        node_membership[best_node_to_switch] = best_class_to_switch_to
        (log_lik,
         fitted_params) = bhm_estimate_utils.estimate_hawkes_param_and_calc_log_likelihood(event_dict,
                                                                                           node_membership,
                                                                                           duration, n_classes,
                                                                                           False)

        if iter == max_iter - 1:
            print("Warning: Max iter reached!")

    if verbose:
        print(f"likelihood went from {init_log_lik:.4f} to {log_lik:.4f}. "
              f"{100 * np.abs((log_lik - init_log_lik) / init_log_lik):.2f}% increase.")

    if return_fitted_param:
        mu, alpha, beta = fitted_params
        return node_membership, mu, alpha, beta

    return node_membership


# Example of running local search on Block Hawkes model.
if __name__ == '__main__':
    seed = None
    n_classes = 4
    n_nodes = 64
    duration = 50
    class_probs = np.ones(n_classes) / n_classes

    alpha = 0.6
    beta = 0.8
    mu_diag = 1.6
    mu_off_diag = 0.8

    bp_alpha = np.ones((n_classes, n_classes), dtype=np.float) * alpha
    bp_beta = np.ones((n_classes, n_classes), dtype=np.float) * beta
    bp_mu = np.ones((n_classes, n_classes), dtype=np.float) * mu_off_diag
    np.fill_diagonal(bp_mu, mu_diag)

    true_class_assignments, event_dict = bhm.block_generative_model(n_nodes, class_probs,
                                                                    bp_mu, bp_alpha, bp_beta,
                                                                    duration, seed=seed)
    true_class_assignments = utils.one_hot_to_class_assignment(true_class_assignments)

    binary_adj = utils.event_dict_to_adjacency(n_nodes, event_dict)
    spectral_node_membership = regularized_spectral_cluster(binary_adj, num_classes=n_classes)

    sc_rand = adjusted_rand_score(true_class_assignments, spectral_node_membership)
    print(f"SC Rand index: {sc_rand:.3f}")

    print("Parallel")
    tic = time.time()
    local_search_node_membership = block_local_search(event_dict, n_classes, spectral_node_membership, duration,
                                                      max_iter=10, n_cores=34, verbose=True)
    toc = time.time()
    print(f"local search took {toc - tic:.2f}s.")

    sc_rand = adjusted_rand_score(true_class_assignments, local_search_node_membership)
    print(f"Local search Rand index: {sc_rand:.3f}")

