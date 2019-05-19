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
from community_generative_model import community_generative_model
from spectral_clustering import spectral_cluster
from sklearn.metrics import adjusted_rand_score

#%% Set parameter values
number_of_nodes = 128
class_probabilities = [.25, .25, .25, .25]
end_time = 300
num_of_classes = len(class_probabilities)

#bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 7500
#bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 8000
bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.8
#bp_alpha = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 5000
#np.fill_diagonal(bp_alpha, 15000)
#bp_beta = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 5400
#np.fill_diagonal(bp_beta, 16200)
bp_mu = np.ones((num_of_classes, num_of_classes), dtype=np.float) * 0.6
np.fill_diagonal(bp_mu, 1.8)

bp_mu = utils.scale_parameteres_by_block_pair_size(bp_mu, 128,
                                                   class_probabilities)
bp_alpha = utils.scale_parameteres_by_block_pair_size(bp_alpha, 128,
                                                      class_probabilities)
bp_beta = utils.scale_parameteres_by_block_pair_size(bp_beta, 128,
                                                     class_probabilities)

#%% Generate data
node_membership, event_dicts = community_generative_model(
        number_of_nodes,class_probabilities,bp_mu,bp_alpha,bp_beta,end_time)
true_class_assignments = utils.one_hot_to_class_assignment(node_membership)
adj = utils.event_dict_to_adjacency(number_of_nodes, event_dicts)
agg_adj = utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts)

#%% Perform spectral clustering
adj_sc_pred = spectral_cluster(adj, num_classes=num_of_classes)
agg_adj_pred = spectral_cluster(agg_adj, num_classes=num_of_classes)

#%% Check cluster sizes
print('True cluster sizes:')
print(np.histogram(true_class_assignments,range(num_of_classes+1)))
print('Unweighted spectral clustering cluster sizes:')
print(np.histogram(adj_sc_pred,range(num_of_classes+1)))
print('Weighted spectral clustering cluster sizes:')
print(np.histogram(agg_adj_pred,range(num_of_classes+1)))

#%% Check agreement with true clusters
print('Unweighted spectral clustering ARI:')
adj_sc_rand = adjusted_rand_score(true_class_assignments, adj_sc_pred)
print(adj_sc_rand)
print('Weighted spectral clustering ARI:')
agg_adj_sc_rand = adjusted_rand_score(true_class_assignments, agg_adj_pred)
print(agg_adj_sc_rand)

#%% Check agreement with node degrees
out_deg_adj = np.sum(adj,axis=0)
in_deg_adj = np.sum(adj,axis=1)
out_deg_agg_adj = np.sum(agg_adj,axis=0)
in_deg_agg_adj = np.sum(agg_adj,axis=1)

# Sort spectral clustering result by cluster number and output node degrees
sc_sort_idx = np.argsort(adj_sc_pred)
print(np.r_[adj_sc_pred[np.newaxis,sc_sort_idx],
            out_deg_adj[np.newaxis,sc_sort_idx],
            in_deg_adj[np.newaxis,sc_sort_idx]])
plt.figure()
plt.subplot(2,1,1)
plt.title('Unweighted spectral clustering')
plt.plot(adj_sc_pred[sc_sort_idx])
plt.subplot(2,1,2)
plt.plot(out_deg_adj[sc_sort_idx])
plt.plot(in_deg_adj[sc_sort_idx])

agg_sc_sort_idx = np.argsort(agg_adj_pred)
print(np.r_[agg_adj_pred[np.newaxis,agg_sc_sort_idx],
            out_deg_agg_adj[np.newaxis,agg_sc_sort_idx],
            in_deg_agg_adj[np.newaxis,agg_sc_sort_idx]])
plt.figure()
plt.subplot(2,1,1)
plt.title('Weighted spectral clustering')
plt.plot(agg_adj_pred[agg_sc_sort_idx])
plt.subplot(2,1,2)
plt.plot(out_deg_agg_adj[agg_sc_sort_idx])
plt.plot(in_deg_agg_adj[agg_sc_sort_idx])
