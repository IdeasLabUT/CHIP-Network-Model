import time
import numpy as np
import dataset_utils
import parameter_estimation as estimate_utils
import generative_model_utils as utils
from spectral_clustering import spectral_cluster


num_classes = 100

tic = time.time()
fb_event_dict, fb_num_node, fb_duration = dataset_utils.load_facebook_wall(largest_connected_component_only=True)
toc = time.time()
print(f"Loaded the dataset in {toc - tic:.1f}s")

print("Num Nodes:", fb_num_node, "Duration:", fb_duration,
      "Num Edges:", utils.num_events_in_event_dict(fb_event_dict))

tic = time.time()
agg_adj = utils.event_dict_to_aggregated_adjacency(fb_num_node, fb_event_dict)
adj = utils.event_dict_to_adjacency(fb_num_node, fb_event_dict)
toc = time.time()
print(f"Generated aggregated adj in {toc - tic:.1f}s")

tic = time.time()
# Running spectral clustering
node_membership = spectral_cluster(adj, num_classes=num_classes, verbose=True, plot_eigenvalues=True)
toc = time.time()
print(f"Spectral clustering done in {toc - tic:.1f}s")
print("Community assignment prob:", np.unique(node_membership, return_counts=True)[1] / fb_num_node)

tic = time.time()
bp_mu, bp_alpha_beta_ratio = estimate_utils.estimate_hawkes_from_counts(agg_adj, node_membership,
                                                                        fb_duration,
                                                                        1e-10 / fb_duration)
toc = time.time()
print(f"Mu and alpha estimated in {toc - tic:.1f}s")
#
# print("Mu:")
# print(bp_mu)
# print("Ratio:")
# print(bp_alpha_beta_ratio)

# print("\nStart Beta estimation:")
# tic = time.time()
# bp_beta = np.zeros((num_classes, num_classes), dtype=np.float)
# block_pair_events = utils.event_dict_to_block_pair_events(fb_event_dict, node_membership, num_classes)
#
# cnt = 0
# for b_i in range(num_classes):
#     for b_j in range(num_classes):
#         bp_size = len(np.where(node_membership == b_i)[0]) * len(np.where(node_membership == b_j)[0])
#         if b_i == b_j:
#             bp_size -= len(np.where(node_membership == b_i)[0])
#
#         bp_beta[b_i, b_j], _ = estimate_utils.estimate_beta_from_events(block_pair_events[b_i][b_j],
#                                                                         bp_mu[b_i, b_j],
#                                                                         bp_alpha_beta_ratio[b_i, b_j],
#                                                                         fb_duration, bp_size)
#         cnt += 1
#         print(f"{100 * cnt / num_classes ** 2:0.2f}% Done.", end='\r')
#
# bp_alpha = bp_alpha_beta_ratio * bp_beta
# toc = time.time()
# print(f"Beta estimated in {toc - tic:.1f}s")
#
# print("Alpha")
# print(bp_alpha)
#
# print("Beta")
# print(bp_beta)

