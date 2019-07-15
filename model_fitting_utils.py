import numpy as np
import dataset_utils
import chp_local_search as cls
import matplotlib.pyplot as plt
from scipy.stats import multinomial
import generative_model_utils as utils
import parameter_estimation as estimate_utils
from spectral_clustering import spectral_cluster
from chp_model_fitting import fit_and_eval_community_hawkes
from community_generative_model import community_generative_model
from block_generative_model import block_generative_model


def fit_community_model(event_dict, num_nodes, duration, num_classes, local_search_max_iter, local_search_n_cores,
                        verbose=False):
    agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)
    # adj = utils.event_dict_to_adjacency(num_nodes, event_dict)

    # Running spectral clustering
    node_membership = spectral_cluster(agg_adj, num_classes, verbose=False)

    if local_search_max_iter > 0 and num_classes > 1:
        node_membership, bp_mu, bp_alpha, bp_beta = cls.chp_local_search(event_dict, num_classes, node_membership,
                                                                         duration,
                                                                         max_iter=local_search_max_iter,
                                                                         n_cores=local_search_n_cores,
                                                                         return_fitted_param=True, verbose=False)

        block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, num_classes)

    else:
        bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                                duration,
                                                                                1e-10 / duration)
        bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)

        block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, num_classes)

        for b_i in range(num_classes):
            for b_j in range(num_classes):
                bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
                if b_i == b_j:
                    bp_size -= len(np.where(node_membership == b_i)[0])

                bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
                                                                                bp_mu[b_i, b_j],
                                                                                bp_alpha_beta_ratio[b_i, b_j],
                                                                                duration, bp_size)

        bp_alpha = bp_alpha_beta_ratio * bp_beta

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

    return node_membership, bp_mu, bp_alpha, bp_beta, block_pair_events


def estimate_bp_hawkes_params(event_dict, node_membership, duration, num_classes):
    num_nodes = len(node_membership)

    agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)
    bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                            duration,
                                                                            1e-10 / duration)

    bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)
    block_pair_events = utils.event_dict_to_block_pair_events(event_dict, node_membership, num_classes)

    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
                                                                            bp_mu[b_i, b_j],
                                                                            bp_alpha_beta_ratio[b_i, b_j],
                                                                            duration, bp_size)

    bp_alpha = bp_alpha_beta_ratio * bp_beta

    return bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio


def calc_full_log_likelihood(block_pair_events, node_membership,
                             bp_mu, bp_alpha, bp_beta,
                             duration, num_classes,
                             add_com_assig_log_prob=True):
    log_likelihood = 0
    for b_i in range(num_classes):
        for b_j in range(num_classes):
            bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
            if b_i == b_j:
                bp_size -= len(np.where(node_membership == b_i)[0])

            log_likelihood += estimate_utils.block_pair_full_hawkes_log_likelihood(block_pair_events[b_i][b_j],
                                                                                   bp_mu[b_i, b_j], bp_alpha[b_i, b_j],
                                                                                   bp_beta[b_i, b_j], duration,
                                                                                   block_pair_size=bp_size)

    if add_com_assig_log_prob:
        # Adding the log probability of the community assignments to the full log likelihood
        n_nodes = len(node_membership)
        _, block_count = np.unique(node_membership, return_counts=True)
        class_prob_mle = block_count / sum(block_count)
        rv_multi = multinomial(n_nodes, class_prob_mle)
        log_prob_community_assignment = rv_multi.logpmf(block_count)

        log_likelihood += log_prob_community_assignment

    return log_likelihood


def assign_node_membership_for_missing_nodes(node_membership, missing_nodes):
    class_idx, class_count = np.unique(node_membership, return_counts=True)
    largest_class_idx = class_idx[np.argmax(class_count)]

    combined_node_membership = np.copy(node_membership)

    missing_nodes.sort()
    for n in missing_nodes:
        combined_node_membership = np.insert(combined_node_membership, n, largest_class_idx)

    return combined_node_membership


def calc_per_event_log_likelihood(combined_log_likelihood, train_log_likelihood, test_event_dict, test_num_nodes):
    test_num_events = np.sum(utils.event_dict_to_aggregated_adjacency(test_num_nodes, test_event_dict))
    return (combined_log_likelihood - train_log_likelihood) / test_num_events


def generate_fit_community_hawkes(event_dict, node_membership,
                                  bp_mu, bp_alpha, bp_beta,
                                  duration, fit_generated_model, plot_hist, train_percentage=0.8, n_cores=1,
                                  seed=None):
    """
    Generates a community model the plots its count histogram against the original event_dict. (if plot_hist is True)

    Also, fits a community model to the generated data itself (if fit_generated_model is True), this is done by
    converting the generated event_dict to event_list and split it to train\test and fit it using model_fitting.
    """
    if not fit_generated_model and not plot_hist:
        return

    # Generating a network
    n_nodes = len(node_membership)

    _, block_count = np.unique(node_membership, return_counts=True)
    class_prob = block_count / sum(block_count)

    generated_node_membership, generated_event_dict = community_generative_model(n_nodes, class_prob,
                                                                                 bp_mu, bp_alpha, bp_beta,
                                                                                 burnin=None, end_time=duration,
                                                                                 n_cores=n_cores, seed=seed)

    if plot_hist:
        generated_agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, generated_event_dict, dtype=np.int)
        generated_deg_count_flattened = np.reshape(generated_agg_adj, (n_nodes * n_nodes))

        agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict, dtype=np.int)
        deg_count_flattened = np.reshape(agg_adj, (n_nodes * n_nodes))

        plt.hist(deg_count_flattened, bins=50, alpha=0.5, label='Real Data', color='blue', density=True)
        plt.hist(generated_deg_count_flattened, bins=50, alpha=0.5, label='Generated Data', color='red', density=True)

        plt.legend(loc='upper right')
        plt.xlabel('Event Count')
        plt.ylabel('Density')
        plt.title(f'Histogram of the Count Matrix Real Vs. Generated CHP Model Data - K: {len(class_prob)}'
                  f'\n Mean Count -  Real: {np.mean(agg_adj):.3f} - Generated: {np.mean(generated_agg_adj):.3f}')
        plt.yscale("log")
        plt.show()

    if fit_generated_model:
        print("Generated model test ll:")
        gen_event_list = utils.event_dict_to_event_list(generated_event_dict)
        gen_train_tuple, gen_test_tuple, gen_combined_tuple, gen_nodes_not_in_train = \
            dataset_utils.split_event_list_to_train_test(gen_event_list, train_percentage)

        # print(utils.num_events_in_event_dict(ge))
        fit_and_eval_community_hawkes(gen_train_tuple, gen_test_tuple, gen_combined_tuple, gen_nodes_not_in_train)

    return


def generate_fit_block_hawkes(event_dict, node_membership,
                              bp_mu, bp_alpha, bp_beta,
                              duration, seed=None):

    """
    Generates a block model the plots its count histogram against the original event_dict.
    """

    # Generating a network
    n_nodes = len(node_membership)

    _, block_count = np.unique(node_membership, return_counts=True)
    class_prob = block_count / sum(block_count)

    generated_node_membership, generated_event_dict = block_generative_model(n_nodes, class_prob,
                                                                             bp_mu, bp_alpha, bp_beta,
                                                                             end_time=duration, seed=seed)

    generated_agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, generated_event_dict, dtype=np.int)
    generated_deg_count_flattened = np.reshape(generated_agg_adj, (n_nodes * n_nodes))

    agg_adj = utils.event_dict_to_aggregated_adjacency(n_nodes, event_dict, dtype=np.int)
    deg_count_flattened = np.reshape(agg_adj, (n_nodes * n_nodes))

    plt.hist(deg_count_flattened, bins=30, alpha=0.5, label='Real Data', color='blue', density=True)
    plt.hist(generated_deg_count_flattened, bins=30, alpha=0.5, label='Generated Data', color='red', density=True)

    plt.legend(loc='upper right')
    plt.xlabel('Event Count')
    plt.ylabel('Density')
    plt.title(f'Histogram of the Count Matrix Real Vs. Generated Block Model Data - K: {len(class_prob)}'
              f'\n Mean Count -  Real: {np.mean(agg_adj):.3f} - Generated: {np.mean(generated_agg_adj):.3f}')
    plt.yscale("log")
    plt.show()

    return
