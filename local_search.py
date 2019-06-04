import numpy as np
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import spectral_cluster
import parameter_estimation as estimate_utils
import generative_model_utils as utils
import community_model_fitting_utils as fit_utils


def chp_local_search(event_dict, n_classes, node_membership_init, duration, max_iter=100, verbose=True):
    n_nodes = len(node_membership_init)
    node_membership = node_membership_init
    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict, dtype=np.int)

    # estimate initial params of CHP and its log-likelihood
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
                                                                                              default_mu=1e-10/duration)
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

        # if a good neighbor was found, update all CHP params, and go for the next iteration.
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
              f"{100 * np.abs(log_lik - init_log_lik) / init_log_lik:.2f}% increase.")

    return node_membership


if __name__ == '__main__':
    n_classes = 4
    n_nodes = 64
    duration = 800

    params = {'number_of_nodes': n_nodes,
              'alpha': 0.6,
              'beta': 0.8,
              'mu_off_diag': 0.8,
              'mu_diag': 1.6,
              'end_time': duration,
              'class_probabilities': np.ones(n_classes) / n_classes,
              'n_cores': 1}

    event_dict, true_class_assignments = utils.simulate_community_hawkes(params,
                                                                         network_name="local_seach_test_networks",
                                                                         load_if_exists=True)

    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict)
    spectral_node_membership = spectral_cluster(agg_adj, num_classes=n_classes)
    sc_rand = adjusted_rand_score(true_class_assignments, spectral_node_membership)
    print(f"SC Rand index: {sc_rand:.3f}")

    local_search_node_membership = chp_local_search(event_dict, n_classes, spectral_node_membership, duration,
                                                    max_iter=20, verbose=True)

    sc_rand = adjusted_rand_score(true_class_assignments, local_search_node_membership)
    print(f"Local search Rand index: {sc_rand:.3f}")