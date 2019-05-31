import time
import numpy as np
import dataset_utils
import multiprocessing
from joblib import Parallel, delayed
import generative_model_utils as utils


def base_community_generative_model(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta,
                                    node_theta,
                                    burnin, end_time, seed=None):
    """
    Base line community generative model for all variants of the model

    :param num_nodes: (int) Total number of nodes
    :param class_prob: (list) Probability of class memberships from class 0 to K - 1
    :param bp_mu: K x K matrix where entry ij denotes the mu of Hawkes process for block pair (b_i, b_j)
    :param bp_alpha: K x K matrix where entry ij denotes the alpha of Hawkes process for block pair (b_i, b_j)
    :param bp_beta: K x K matrix where entry ij denotes the beta of Hawkes process for block pair (b_i, b_j)
    :param node_theta: list of num_nodes theta values for each node for the degree corrected model. If None, the model
                        will be based on Eq 2.1, else Eq 2.2.
    :param burnin: (int) time before which all events are discarded. None if no burnin needed.
    :param end_time: end_time of hawkes simulation
    :param seed: seed of all random processes

    :return: `node_membership`: membership of every node to one of K classes. num_nodes x num_classes (one_hot)
         `event_dict`: dictionary of (u, v): [event time stamps]
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

                    # select mu based on the model
                    mu = bp_mu[c_i, c_j] if node_theta is None else bp_mu[c_i, c_j] * node_theta[b_i] * node_theta[b_j]

                    event_times = utils.simulate_univariate_hawkes(mu,
                                                                   bp_alpha[c_i, c_j],
                                                                   bp_beta[c_i, c_j],
                                                                   end_time, seed=hawkes_seed)
                    if burnin is not None:
                        for burnin_idx in range(len(event_times)):
                            if event_times[burnin_idx] >= burnin:
                                event_times = event_times[burnin_idx:]
                                break
                        else:
                            event_times = np.array([])

                    if len(event_times) > 0:
                        event_dicts[(b_i, b_j)] = event_times

    return node_membership, event_dicts


def base_community_generative_model_parallel(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta,
                                             node_theta,
                                             burnin, end_time, n_cores=-1, seed=None):
    """
    Base line community generative model for all variants of the model


    :param num_nodes: (int) Total number of nodes
    :param class_prob: (list) Probability of class memberships from class 0 to K - 1
    :param bp_mu: K x K matrix where entry ij denotes the mu of Hawkes process for block pair (b_i, b_j)
    :param bp_alpha: K x K matrix where entry ij denotes the alpha of Hawkes process for block pair (b_i, b_j)
    :param bp_beta: K x K matrix where entry ij denotes the beta of Hawkes process for block pair (b_i, b_j)
    :param node_theta: list of num_nodes theta values for each node for the degree corrected model. If None, the model
                        will be based on Eq 2.1, else Eq 2.2.
    :param burnin: (int) time before which all events are discarded. None if no burnin needed.
    :param end_time: end_time of hawkes simulation
    :param n_cores: number of parallel cores to be used. If -1, maximum number of cores will be used.
    :param seed: seed of all random processes

    :return: `node_membership`: membership of every node to one of K classes. num_nodes x num_classes (one_hot)
         `event_dict`: dictionary of (u, v): [event time stamps]
    """

    np.random.seed(seed)

    num_classes = len(class_prob)
    node_membership, community_membership = utils.assign_class_membership(num_nodes, class_prob)

    event_dicts = {}

    hawkes_seed = seed
    n_cores = n_cores if n_cores > 0 else multiprocessing.cpu_count()

    for c_i in range(num_classes):
        if len(community_membership[c_i]) == 0:
            continue

        for c_j in range(num_classes):
            if len(community_membership[c_j]) == 0:
                continue

            bp_event_dicts = Parallel(n_jobs=n_cores)(delayed(generate_hawkes_for_single_node_block_pair)
                                                             (bp_mu, bp_alpha, bp_beta, node_theta,
                                                              burnin, end_time, hawkes_seed,
                                                              c_i, c_j, b_i, community_membership[c_j])
                                                      for b_i in community_membership[c_i])

            for bp_event_dict in bp_event_dicts:
                event_dicts.update(bp_event_dict)

    return node_membership, event_dicts


def generate_hawkes_for_single_node_block_pair(bp_mu, bp_alpha, bp_beta,
                                               node_theta,
                                               burnin, end_time, seed, c_i, c_j, b_i, b_js):
    bp_event_dict = {}

    hawkes_seed = seed
    for b_j in b_js:
        # self events are not allowed
        if b_i == b_j:
            continue

        # Seed has to change in order to get different event times for each node pair.
        hawkes_seed = None if seed is None else hawkes_seed + 1

        # select mu based on the model
        mu = bp_mu[c_i, c_j] if node_theta is None else bp_mu[c_i, c_j] * node_theta[b_i] * node_theta[b_j]

        event_times = utils.simulate_univariate_hawkes(mu,
                                                       bp_alpha[c_i, c_j],
                                                       bp_beta[c_i, c_j],
                                                       end_time, seed=hawkes_seed)
        if burnin is not None:
            for burnin_idx in range(len(event_times)):
                if event_times[burnin_idx] >= burnin:
                    event_times = event_times[burnin_idx:]
                    break
            else:
                event_times = np.array([])

        if len(event_times) > 0:
            bp_event_dict[(b_i, b_j)] = event_times

    return bp_event_dict


def community_generative_model(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta, burnin, end_time, n_cores=1, seed=None):
    """
    This is based on Eq 2.1

    Check doc string of `base_community_generative_model` or `base_community_generative_model_parallel`.
    """
    if n_cores == 1:
        return base_community_generative_model(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta,
                                               node_theta=None, burnin=burnin, end_time=end_time, seed=seed)

    return base_community_generative_model_parallel(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta,
                                                    node_theta=None, burnin=burnin, end_time=end_time,
                                                    n_cores=n_cores, seed=None)


def degree_corrected_community_generative_model(num_nodes, class_prob,
                                                bp_mu, bp_alpha, bp_beta,
                                                node_theta,
                                                burnin, end_time, n_cores=1, seed=None):
    """
    This is based on Eq 2.2

    Check doc string of `base_community_generative_model` or `base_community_generative_model_parallel`.
    """
    if n_cores == 1:
        return base_community_generative_model(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta,
                                               node_theta=node_theta, burnin=burnin, end_time=end_time, seed=seed)

    return base_community_generative_model_parallel(num_nodes, class_prob, bp_mu, bp_alpha, bp_beta,
                                                    node_theta=node_theta, burnin=burnin, end_time=end_time,
                                                    n_cores=n_cores, seed=None)


if __name__ == "__main__":
    seed = None
    number_of_nodes = 1000
    class_probabilities = [0.25, 0.25, 0.25, 0.25]
    num_of_classes = len(class_probabilities)
    end_time = 400
    burnin = None

    bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 7.500
    bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 8.000
    bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
    np.fill_diagonal(bp_mu, 1.8)

    bp_mu = utils.scale_parameteres_by_block_pair_size(bp_mu, 128, class_probabilities)
    bp_alpha = utils.scale_parameteres_by_block_pair_size(bp_alpha, 128, class_probabilities)
    bp_beta = utils.scale_parameteres_by_block_pair_size(bp_beta, 128, class_probabilities)

    tic = time.time()
    node_membership, event_dicts = community_generative_model(number_of_nodes,
                                                              class_probabilities,
                                                              bp_mu, bp_alpha, bp_beta,
                                                              burnin, end_time, seed=seed)
    toc = time.time()

    print(toc - tic)

    tic = time.time()
    node_memberships, event_dictss = community_generative_model(number_of_nodes,
                                                              class_probabilities,
                                                              bp_mu, bp_alpha, bp_beta,
                                                              burnin, end_time, n_cores=-1, seed=seed)
    toc = time.time()
    print(toc - tic)

    exit()

    node_membership = utils.one_hot_to_class_assignment(node_membership)

    block_pair_events = utils.event_dict_to_block_pair_events(event_dicts, node_membership, num_of_classes)
    print(block_pair_events)
    exit()

    # theta = utils.generate_theta_params_for_degree_corrected_community(number_of_nodes, dist='dirichlet', norm_sum_to='n')
    #
    # node_membership, event_dicts = degree_corrected_community_generative_model(number_of_nodes,
    #                                                                            class_probabilities,
    #                                                                            bp_mu, bp_alpha, bp_beta,
    #                                                                            theta,
    #                                                                            burnin, end_time, seed=seed)
    #
    # dataset_utils.plot_event_count_hist(event_dicts, number_of_nodes, "DC Community Hawkes")

    # Check if the theoretical mean gets closer to empirical by scaling T and Mu

    # for s in [1, 2, 3, 4]:
    for s in [1]:
        print("scalar", s)
        end_time = 150 * s
        burnin=100

        # bp_mu, bp_alpha, bp_beta = utils.generate_random_hawkes_params(num_of_classes,
        #                                                                mu_range=(0.1, 0.3),
        #                                                                alpha_range=(0.2, 0.4),
        #                                                                beta_range=(0.5, 1),
        #                                                                seed=seed)

        bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 7500
        bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 8000
        bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6 / s
        np.fill_diagonal(bp_mu, 1.8 / s)

        bp_mu = utils.scale_parameteres_by_block_pair_size(bp_mu, 128, class_probabilities)
        bp_alpha = utils.scale_parameteres_by_block_pair_size(bp_alpha, 128, class_probabilities)
        bp_beta = utils.scale_parameteres_by_block_pair_size(bp_beta, 128, class_probabilities)

        # print(bp_mu)
        # print(bp_alpha)
        # print(bp_beta)
        #
        # m = (bp_mu * end_time) / (1 - (bp_alpha/bp_beta))
        #
        # print(m)
        # print(np.mean(m))

        event_count_means = []

        for i in range(100):
            node_membership, event_dicts = community_generative_model(number_of_nodes,
                                                                      class_probabilities,
                                                                      bp_mu, bp_alpha, bp_beta,
                                                                      burnin, end_time, seed=seed)

            # dataset_utils.plot_event_count_hist(event_dicts, number_of_nodes, "Community Hawkes")
            event_agg_adj = utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts, dtype=np.int)

            # np.savetxt(f"community-hawkes-{i}.txt", event_agg_adj, delimiter=' ', fmt='%d')

            num_events = np.reshape(event_agg_adj, number_of_nodes**2)

            event_count_means.append(np.mean(num_events))

        print("mean:", np.mean(event_count_means))
        print("95% Error:", 2 * np.std(event_count_means) / np.sqrt(len(event_count_means)))

    # print(node_membership, event_dicts.keys())
    # print(utils.event_dict_to_adjacency(number_of_nodes, event_dicts))
    # print(utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts))
