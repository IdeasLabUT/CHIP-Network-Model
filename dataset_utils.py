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

            # if rm_data[i, 0] > 1096588800 and rm_data[i, 0] < 1104537600:
            #     cnt += 1

            if (rm_data[i, 1], rm_data[i, 2]) not in event_dict:
                event_dict[(rm_data[i, 1], rm_data[i, 2])] = []

            event_dict[(rm_data[i, 1], rm_data[i, 2])].append(rm_data[i, 0])

    return event_dict, len(core_nodes_id)


def load_reality_mining_test_train():
    """
    TODO: This code is not complete due to issues with DuBios's paper.

    Loads the pre-split dataset for test and train of reality mining.
    :return: (dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure)
             (int) number of nodes
    """

    train_file_path = '/shared/DataSets/RealityMining/Dubois2013/train_reality.csv'
    test_file_path = '/shared/DataSets/RealityMining/Dubois2013/test_reality.csv'

    # load the core dataset. caller_id,receiver_id,unix_timestamp
    train_data = np.loadtxt(train_file_path, np.float, delimiter=',', usecols=(0, 1, 2))
    # Sorting by unix_timestamp
    train_data = train_data[train_data[:, 2].argsort()]
    train_nodes = set()

    train_event_dict = {}
    for i in range(train_data.shape[0]):
        caller_id = np.int(train_data[i, 0])
        receiver_id = np.int(train_data[i, 1])

        train_nodes.add(caller_id)
        train_nodes.add(receiver_id)

        if (caller_id, receiver_id) not in train_event_dict:
            train_event_dict[(caller_id, receiver_id)] = []

        train_event_dict[(caller_id, receiver_id)].append(train_data[i, 0])


    # load the core dataset. caller_id,receiver_id,unix_timestamp
    test_data = np.loadtxt(test_file_path, np.float, delimiter=',', usecols=(0, 1, 2))
    # Sorting by unix_timestamp
    test_data = test_data[test_data[:, 2].argsort()]
    test_nodes = set()

    test_event_dict = {}
    for i in range(test_data.shape[0]):
        caller_id = np.int(test_data[i, 0])
        receiver_id = np.int(test_data[i, 1])

        test_nodes.add(caller_id)
        test_nodes.add(receiver_id)

        if (caller_id, receiver_id) not in test_event_dict:
            test_event_dict[(caller_id, receiver_id)] = []

        test_event_dict[(caller_id, receiver_id)].append(test_data[i, 0])

    print(test_nodes)
    print(len(test_nodes))

    print(train_nodes)
    print(len(train_nodes))

    print(test_nodes.difference(train_nodes))
    print(train_nodes.difference(test_nodes))

    cnt = 0
    for n in test_nodes.difference(train_nodes):
        for u, v in test_event_dict:
            if n == u or n == v:
                cnt += len(test_event_dict[(u, v)])

    print(cnt)
    print(test_data.shape[0])
    print(train_data.shape[0])

    print(test_data.shape[0] + train_data.shape[0])


def load_enron():
    """
    Loads Enron dataset.
    :return: (dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure)
             (int) number of nodes
    """
    edges_file_path = '/shared/DataSets/EnronPriebe2009/raw/execs.email.lines2.txt'

    # load the core dataset.  time, from, receiver, tag
    enron_data = np.loadtxt(edges_file_path, np.int, delimiter=' ', usecols=(0, 1, 2))
    # Sorting by unix_timestamp
    enron_data = enron_data[enron_data[:, 0].argsort()]

    people = set(enron_data[:, 1])
    people = people.union(enron_data[:, 2])

    event_dict = {}
    for i in range(enron_data.shape[0]):
        if (enron_data[i, 1], enron_data[i, 2]) not in event_dict:
            event_dict[(enron_data[i, 1], enron_data[i, 2])] = []

        event_dict[(enron_data[i, 1], enron_data[i, 2])].append(enron_data[i, 0])

    print(event_dict)
    return event_dict, len(people)


def plot_event_count_hist(event_dict, num_nodes, dset_title_name):
    """
    Plot Histogram of Event Count
    :param event_dict: event_dict of interactions
    :param num_nodes: number of nodes in the dataset
    :param dset_title_name: Name of the dataset to be added to the title
    :rtype: None (show hist)
    """
    event_agg_adj = utils.event_dict_to_aggregated_adjacency(num_nodes, event_dict)

    num_events = np.reshape(event_agg_adj, num_nodes**2)

    plt.hist(num_events, 50, density=True)
    plt.xlabel("Number of Events")
    plt.ylabel("Density")
    plt.title(f"Histogram of {dset_title_name}'s Number of Interactions \n"
              f" Mean Count: {np.mean(num_events):.4f}, Total count: {np.sum(num_events)}")
    plt.yscale("log")
    plt.show()


if __name__ == '__main__':
    # reality_mining_event_dict, num_nodes = load_core_reality_mining()
    # plot_event_count_hist(reality_mining_event_dict, num_nodes, "Reality Mining's Core people")

    # enron_event_dict, num_nodes = load_enron()
    # plot_event_count_hist(enron_event_dict, num_nodes, "Enron")
    # print(load_enron())

    load_reality_mining_test_train()
    load_core_reality_mining()
