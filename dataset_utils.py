import numpy as np
import matplotlib.pyplot as plt
import generative_model_utils as utils


def load_core_reality_mining():
    """
    Loads only the interaction of the core people of the reality mining dataset.
    :return: (dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure)
             (int) number of nodes
    """
    attributes_file_path = '/shared/DataSets/RealityMining/Dubois2013/all.attributes.txt'
    edges_file_path = '/shared/DataSets/RealityMining/Dubois2013/voice-all.edges'

    # get the id of core nodes from all.attribute. Each line is 'node_id';'1' or '0' if it is in core or not
    core_nodes_id = set()
    with open(attributes_file_path, 'r') as f:
        for l in f:
            node_id, is_core = l.split(';')

            if int(is_core):
                core_nodes_id.add(int(node_id))

    # load the core dataset. unix_timestamp;caller_id;receiver_id;duration_in_second;communication_type
    rm_data = np.loadtxt(edges_file_path, np.int, delimiter=';', usecols=(0, 1, 2))
    # Sorting by unix_timestamp
    rm_data = rm_data[rm_data[:, 0].argsort()]

    event_dict = {}
    for i in range(rm_data.shape[0]):
        if rm_data[i, 1] in core_nodes_id and rm_data[i, 2] in core_nodes_id:
            if (rm_data[i, 1], rm_data[i, 2]) not in event_dict:
                event_dict[(rm_data[i, 1], rm_data[i, 2])] = []

            event_dict[(rm_data[i, 1], rm_data[i, 2])].append(rm_data[i, 0])

    return event_dict, len(core_nodes_id)


def plot_reality_mining_num_events_hist():
    reality_mining_event_dict, num_nodes = load_core_reality_mining()
    reality_mining_agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, reality_mining_event_dict)

    num_events = np.reshape(reality_mining_agg_adj, num_nodes**2)
    print(num_events)
    print(np.sum(num_events), min(num_events), max(num_events))

    plt.hist(num_events, 50, density=True)
    plt.xlabel("Number of Events")
    plt.ylabel("Density")
    plt.title(f"Histogram of Reality Mining's Core people's Number of Interactions \n"
              f" Mean: {np.mean(num_events):.4f}, Total count: {np.sum(num_events)}")
    plt.yscale("log")
    plt.show()

    # nun_zero_event_counts = num_events[num_events != 0]
    # plt.hist(nun_zero_event_counts, 50, density=True)
    # plt.xlabel("Number of Events")
    # plt.ylabel("Density")
    # plt.title("Histogram of Reality Mining's Core people's Number of Interactions (excludes zeros) \n  "
    #           "Mean: {np.mean(nun_zero_event_counts)")
    # plt.yscale("log")
    # plt.show()


if __name__ == '__main__':
    plot_reality_mining_num_events_hist()