# -*- coding: utf-8 -*-
"""
Script to test community detection on the community Hawkes process model with
both weighted and unweighted spectral clustering. Looks for correlations with
degree.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import generative_model_utils as utils
from chip_generative_model import community_generative_model
from spectral_clustering import spectral_cluster
from sklearn.metrics import adjusted_rand_score
from parameter_estimation import estimate_hawkes_from_counts, \
    estimate_beta_from_events, estimate_all_from_events

#%% Set parameter values
number_of_nodes = 256
class_probabilities = [.25, .25, .25, .25]
end_time = 200
num_of_classes = len(class_probabilities)

#bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 7500
#bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 8000
#bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
#bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.8
bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 10
bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 20
#bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 5000
#np.fill_diagonal(bp_alpha, 15000)
#bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 5400
#np.fill_diagonal(bp_beta, 16200)
bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
#np.fill_diagonal(bp_mu, 1.2)
np.fill_diagonal(bp_mu, 1.8)

bp_mu = utils.scale_parameteres_by_block_pair_size(bp_mu, 128,
                                                   class_probabilities)
bp_alpha = utils.scale_parameteres_by_block_pair_size(bp_alpha, 128,
                                                      class_probabilities)
bp_beta = utils.scale_parameteres_by_block_pair_size(bp_beta, 128,
                                                     class_probabilities)

#bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 1e-3
#np.fill_diagonal(bp_alpha, 6e-3)
#bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 8e-3
##np.fill_diagonal(bp_beta, 4e-3)
#bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 1e-3

#%% Generate data
print('Generating event matrix')
node_membership, event_dicts = community_generative_model(
        number_of_nodes,class_probabilities,bp_mu,bp_alpha,bp_beta,
        burnin=None,end_time=end_time, seed=1)
true_class_assignments = utils.one_hot_to_class_assignment(node_membership)
adj = utils.event_dict_to_adjacency(number_of_nodes, event_dicts)
agg_adj = utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts)

#%% Perform spectral clustering
print('Performing spectral clustering')
adj_sc_pred = spectral_cluster(adj, num_classes=num_of_classes)
agg_adj_pred = spectral_cluster(agg_adj, num_classes=num_of_classes)

##%% Check cluster sizes
#print('True cluster sizes:')
#print(np.histogram(true_class_assignments,range(num_of_classes+1)))
#print('Unweighted spectral clustering cluster sizes:')
#print(np.histogram(adj_sc_pred,range(num_of_classes+1)))
#print('Weighted spectral clustering cluster sizes:')
#print(np.histogram(agg_adj_pred,range(num_of_classes+1)))
#
#%% Check agreement with true clusters
print('Unweighted spectral clustering ARI:')
adj_sc_rand = adjusted_rand_score(true_class_assignments, adj_sc_pred)
print(adj_sc_rand)
print('Weighted spectral clustering ARI:')
agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)
print(agg_adj_sc_rand)

##%% Check agreement with node degrees
#out_deg_adj = np.sum(adj,axis=0)
#in_deg_adj = np.sum(adj,axis=1)
#out_deg_agg_adj = np.sum(agg_adj,axis=0)
#in_deg_agg_adj = np.sum(agg_adj,axis=1)
#
## Sort spectral clustering result by cluster number and output node degrees
#sc_sort_idx = np.argsort(adj_sc_pred)
#print(np.r_[adj_sc_pred[np.newaxis,sc_sort_idx],
#            out_deg_adj[np.newaxis,sc_sort_idx],
#            in_deg_adj[np.newaxis,sc_sort_idx]])
#plt.figure()
#plt.subplot(2,1,1)
#plt.title('Unweighted spectral clustering')
#plt.plot(adj_sc_pred[sc_sort_idx])
#plt.subplot(2,1,2)
#plt.plot(out_deg_adj[sc_sort_idx])
#plt.plot(in_deg_adj[sc_sort_idx])
#
#agg_sc_sort_idx = np.argsort(agg_adj_pred)
#print(np.r_[agg_adj_pred[np.newaxis,agg_sc_sort_idx],
#            out_deg_agg_adj[np.newaxis,agg_sc_sort_idx],
#            in_deg_agg_adj[np.newaxis,agg_sc_sort_idx]])
#plt.figure()
#plt.subplot(2,1,1)
#plt.title('Weighted spectral clustering')
#plt.plot(agg_adj_pred[agg_sc_sort_idx])
#plt.subplot(2,1,2)
#plt.plot(out_deg_agg_adj[agg_sc_sort_idx])
#plt.plot(in_deg_agg_adj[agg_sc_sort_idx])

# #%% Check accuracy of parameter estimates from counts
# print('Estimating parameters from counts')
# #mu_est,ratio_est = estimate_hawkes_from_counts(agg_adj,agg_adj_pred,end_time)
# mu_est,ratio_est = estimate_hawkes_from_counts(agg_adj,adj_sc_pred,end_time)
# print('Estimated mu:')
# print(mu_est)
# print('Actual mu:')
# print(bp_mu)
# print('Estimated alpha/beta:')
# print(ratio_est)
# print('Actual alpha/beta:')
# print(bp_alpha/bp_beta)
#
# block_pair_events = utils.event_dict_to_block_pair_events(event_dicts, adj_sc_pred, num_of_classes)
# beta_est = np.zeros((num_of_classes,num_of_classes))
# for a in range(num_of_classes):
#     for b in range(num_of_classes):
#         print(f'Estimating beta for block pair ({a},{b})')
#         beta_est[a,b] = estimate_beta_from_events(block_pair_events[a][b],
#                 mu_est[a,b], ratio_est[a,b], end_time)[0]
# alpha_est = ratio_est*beta_est
# print('Estimated beta:')
# print(beta_est)
# print('Actual beta:')
# print(bp_beta)
# print('Estimated alpha:')
# print(alpha_est)
# print('Actual alpha:')
# print(bp_alpha)
#
# #%% Check accuracy of parameter estimates from events
# print('Estimating parameters from events')
# param_est = np.zeros((num_of_classes, num_of_classes, 3))
# for a in range(num_of_classes):
#     for b in range(num_of_classes):
#         print(f'Estimating parameters for block pair ({a},{b})')
#         param_est[a,b,:] = estimate_all_from_events(block_pair_events[a][b],
#                  end_time)[0]
# print('Estimated alpha:')
# print(param_est[:,:,0])
# print('Actual alpha:')
# print(bp_alpha)
# print('Estimated beta:')
# print(param_est[:,:,1])
# print('Actual beta:')
# print(bp_beta)
# print('Estimated mu:')
# print(param_est[:,:,2])
# print('Actual mu:')
# print(bp_mu)
# print('Estimated alpha/beta:')
# print(param_est[:,:,0]/param_est[:,:,1])
# print('Actual alpha/beta:')
# print(bp_alpha/bp_beta)

from scipy.optimize import check_grad
from parameter_estimation import neg_log_likelihood_all, \
    neg_log_likelihood_deriv_all

bp_size = len(np.where(adj_sc_pred == 0)[0]) * len(np.where(adj_sc_pred == 0)[0])
if 0 == 0:
    bp_size -= len(np.where(adj_sc_pred == 0)[0])

block_pair_events = utils.event_dict_to_block_pair_events(event_dicts, adj_sc_pred, num_of_classes)
print('Deviation between numerical gradient and gradient function:')
print(check_grad(neg_log_likelihood_all, neg_log_likelihood_deriv_all, 
                 (1e-2,2e-2,2e-5), block_pair_events[0][0], end_time, None))
