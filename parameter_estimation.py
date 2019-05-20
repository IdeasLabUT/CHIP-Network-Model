# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:34:50 2019

@author: kevin
"""

import numpy as np

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
