import numpy as np
from spectral_clustering import spectral_cluster
from block_generative_model import block_generative_model
from community_generative_model import community_generative_model
import generative_model_utils as utils

# block model
seed = 5
number_of_nodes = 10
class_probabilities = [0.2, 0.4, 0.1, 0.2, 0.1]
num_of_classes = len(class_probabilities)
end_time = 10
bp_mu, bp_alpha, bp_beta = utils.generate_random_hawkes_params(num_of_classes,
                                                               mu_range=(0.1, 0.3),
                                                               alpha_range=(0.2, 0.4),
                                                               beta_range=(0.5, 1),
                                                               seed=seed)

node_membership, event_dicts = community_generative_model(number_of_nodes,
                                                          class_probabilities,
                                                          bp_mu, bp_alpha, bp_beta,
                                                          end_time, seed=seed)

adj = utils.event_dict_to_adjacency(number_of_nodes, event_dicts)
agg_adj = utils.event_dict_to_aggregated_adjacency(number_of_nodes, event_dicts)

# print(adj)
# print(agg_adj)
#
print("Spectral:", spectral_cluster(adj.astype(np.float), num_classes=5))
print("Spectral:", spectral_cluster(agg_adj, num_classes=5))
print("True:", utils.one_hot_to_class_assignment(node_membership))
