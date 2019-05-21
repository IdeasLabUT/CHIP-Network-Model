# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:34:50 2019

@author: kevin
"""

import numpy as np
import tick.hawkes as tick
import generative_model_utils as utils

from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti, HawkesExpKern
import matplotlib.pyplot as plt


def estimate_hawkes_from_counts(agg_adj, class_vec, duration):
    num_classes = class_vec.max()+1
    sample_mean = np.zeros((num_classes,num_classes))
    sample_var = np.zeros((num_classes,num_classes))
    for a in range(num_classes):
        for b in range(num_classes):
            nodes_in_a = np.where(class_vec==a)[0]
            nodes_in_b = np.where(class_vec==b)[0]
            agg_adj_block = agg_adj[nodes_in_a[:,np.newaxis],nodes_in_b]
            if a == b:
                # For diagonal blocks, need to make sure we're not including
                # the diagonal entries of the adjacency matrix in our
                # calculations, so extract indices for the lower and upper
                # triangular portions
                num_nodes_in_a = nodes_in_a.size
                lower_indices = np.tril_indices(num_nodes_in_a,-1)
                upper_indices = np.triu_indices(num_nodes_in_a,1)
                agg_adj_block_no_diag = np.r_[agg_adj_block[lower_indices],
                                              agg_adj_block[upper_indices]]
                sample_mean[a,b] = np.mean(agg_adj_block_no_diag)
                sample_var[a,b] = np.var(agg_adj_block_no_diag,ddof=1)
                
            else:
                sample_mean[a,b] = np.mean(agg_adj_block)
                sample_var[a,b] = np.var(agg_adj_block,ddof=1)
    
    mu = np.sqrt(sample_mean**3 / sample_var) / duration
    alpha_beta_ratio = 1 - np.sqrt(sample_mean / sample_var)
    return (mu, alpha_beta_ratio)


def estimate_hawkes_least_sq(event_times, beta, learner_param_dict=None):

    # Setting up parameters for estimation
    default_params = {'penalty': 'l2',
                      'C': 1,
                      'gofit': 'least-squares',
                      'verbose': True,
                      'tol': 1e-11,
                      'solver': 'bfgs',
                      'step': 1e-3,
                      'max_iter': 1000}

    if learner_param_dict is not None:
        default_params.update(learner_param_dict)

    event_times = [event_times]

    learner = tick.HawkesExpKern(beta, penalty=default_params['penalty'], C=default_params['C'],
                                 gofit=default_params['gofit'], verbose=default_params['verbose'],
                                 tol=default_params['tol'], solver=default_params['solver'],
                                 step=default_params['step'], max_iter=default_params['max_iter'])

    learner.fit(event_times, start=0.1)

    # least_sq_model = tick.hawkes.ModelHawkesExpKernLeastSq(decays=decays, approx=approx, n_threads=-1)

    print("Adj", learner.adjacency)
    print("mu", learner.baseline)

    # a = least_sq_model.fit(event_times, end_times=duration)
    #
    # print(least_sq_model, a)

    # end_time = 10
    # n_realizations = 10
    #
    # decays = 3.
    # baseline = [0.12, 0.07]
    # adjacency = [[.3, 0.], [.6, .21]]
    #
    # hawkes_exp_kernels = SimuHawkesExpKernels(
    #     adjacency=adjacency, decays=decays, baseline=baseline,
    #     end_time=end_time, verbose=False, seed=1039)
    #
    # multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)
    #
    # multi.end_time = [(i + 1) / 10 * end_time for i in range(n_realizations)]
    # multi.simulate()
    #
    # print(multi.timestamps)
    # print(len(multi.timestamps))
    # print(len(multi.timestamps[0]))
    # exit()
    #
    # learner = HawkesExpKern(decays, penalty='l1', C=10, gofit='likelihood',
    #                         verbose=True, tol=1e-11, solver='svrg', step=1e-3)
    # learner.fit(multi.timestamps, start=0.1)
    #
    # # plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels)
    #
    # print(learner.baseline)
    # print(learner.adjacency)

if __name__ == "__main__":

    utils.event_dict_to_block_pair_events(event_dicts, class_assignment)

    mu = 1.5
    alpha = 0.6
    beta = 0.8
    duration = 100

    hawkes_events = utils.simulate_univariate_hawkes(mu, alpha, beta, duration, seed=1)

    estimate_hawkes_least_sq(hawkes_events, beta, duration)
