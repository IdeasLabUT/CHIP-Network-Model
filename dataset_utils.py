import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import generative_model_utils as utils


def load_core_reality_mining_based_on_dubois(remove_nodes_not_in_train=False):
    """
    Due to issues with DoBios' REM paper, this method's network doesn't match the papers. DO NOT USE.

    Loads only the interaction of the core people of the reality mining dataset.

    Loads 2000 phone calls among the 89 recipients between October 2004 and February 2005 (Mtest = 1000).
    The time stamp found to correspond to this description is start from 1098561155 to the end of the entire dataset.

    Of the entire dataset:
    first time stamp: 2004-09-24 00:00:59
    last time stamp: 2005-01-07 23:56:18

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
         ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
         (int) number of nodes,
         (float) duration)
         (list) nodes_not_in_train
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

    # t_from = datetime(2004, 11, 23, 15, 52, 34).timestamp()
    t_from = 1098561155

    recipients_node_id = set()

    # Finding the 89 core recipients
    for i in range(rm_data.shape[0]):
        if rm_data[i, 2] in core_nodes_id:
            recipients_node_id.add(int(rm_data[i, 2]))

    event_list = np.zeros((2000, 3), dtype=np.int)
    core_rec_dict = set()
    cnt = 0
    for i in range(rm_data.shape[0]):
        if t_from <= rm_data[i, 0] and rm_data[i, 1] in recipients_node_id and rm_data[i, 2] in recipients_node_id:
            event_list[cnt] = [rm_data[i, 1], rm_data[i, 2], rm_data[i, 0]]
            cnt += 1

            core_rec_dict.add(int(rm_data[i, 1]))
            core_rec_dict.add(int(rm_data[i, 2]))

    return split_event_list_to_train_test(event_list, train_percentage=0.5,
                                          remove_nodes_not_in_train=remove_nodes_not_in_train)


def load_reality_mining_test_train(remove_nodes_not_in_train):
    """
        Loads Reality Mining dataset.
        :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
                 ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
                 (int) number of nodes,
                 (float) duration)
                 (list) nodes_not_in_train
        """
    train_file_path = '/shared/DataSets/RealityMining/Dubois2013/train_reality.csv'
    test_file_path = '/shared/DataSets/RealityMining/Dubois2013/test_reality.csv'

    # Timestamps are adjusted to start from 0 and go up to 1000.
    combined_duration = 1000.0

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)


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

    return event_dict, len(people)


def load_enron_train_test(remove_nodes_not_in_train=False):
    """
    Loads Enron dataset.
    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = '/shared/DataSets/EnronPriebe2009/train_enron.csv'
    test_file_path = '/shared/DataSets/EnronPriebe2009/test_enron.csv'

    # Timestamps are adjusted to start from 0 and go up to 1000.
    combined_duration = 1000.0

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)


def load_fb_train_test(remove_nodes_not_in_train=False):
    """
    Loads FB dataset.
    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = '/shared/DataSets/FacebookViswanath2009/train_FB_event_mat.csv'
    test_file_path = '/shared/DataSets/FacebookViswanath2009/test_FB_event_mat.csv'

    # Timestamps are adjusted to start from 0 and go up to 8759.9.
    combined_duration = 8759.9

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)


def load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train):
    """
    Loads datasets already split into train and test, such as Enron and FB.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """

    combined_node_id_map, train_node_id_map, test_node_id_map, nodes_not_in_train = \
        load_and_combine_nodes_for_test_train(train_file_path, test_file_path, remove_nodes_not_in_train)

    train_event_dict, train_duration = load_test_train_data(train_file_path, train_node_id_map)
    test_event_dict, test_duration = load_test_train_data(test_file_path, test_node_id_map)
    combined_event_dict = load_test_train_combined(train_file_path, test_file_path, combined_node_id_map)

    return ((train_event_dict, len(train_node_id_map), train_duration),
            (test_event_dict, len(test_node_id_map), test_duration),
            (combined_event_dict, len(combined_node_id_map), combined_duration),
            nodes_not_in_train)


def load_and_combine_nodes_for_test_train(train_path, test_path, remove_nodes_not_in_train):
    """
    Loads the set of nodes in both train and test datasets and maps all the node ids to start form 0 to num total nodes.
    :param train_path
    :param test_path
    :param remove_nodes_not_in_train: if True, all the nodes in test and combined that are not in train, will be removed
    :return `full_node_id_map` dict mapping node id in the entire dataset to a range from 0 to n_full
            `train_node_id_map` dict mapping node id in the train dataset to a range from 0 to n_train
            `test_node_id_map` dict mapping node id in the test dataset to a range from 0 to n_test
            `nodes_not_in_train` list of mapped node ids that are in test, but not in train.
    """

    # load dataset. caller_id,receiver_id,unix_timestamp

    # Train data
    train_nodes = np.loadtxt(train_path, np.int, delimiter=',', usecols=(0, 1))
    train_nodes_set = set(train_nodes.reshape(train_nodes.shape[0] * 2))
    train_node_id_map = get_node_map(train_nodes_set)

    # Test data
    test_nodes = np.loadtxt(test_path, np.int, delimiter=',', usecols=(0, 1))
    test_nodes_set = set(test_nodes.reshape(test_nodes.shape[0] * 2))
    if remove_nodes_not_in_train:
        test_nodes_set = test_nodes_set - test_nodes_set.difference(train_nodes_set)
    test_node_id_map = get_node_map(test_nodes_set)

    # Combined
    if remove_nodes_not_in_train:
        full_node_id_map = train_node_id_map
    else:
        all_nodes = list(train_nodes_set.union(test_nodes_set))
        full_node_id_map = get_node_map(all_nodes)
        all_nodes.sort()

    nodes_not_in_train = []
    for n in test_nodes_set.difference(train_nodes_set):
        nodes_not_in_train.append(full_node_id_map[n])

    return full_node_id_map, train_node_id_map, test_node_id_map, nodes_not_in_train


def get_node_map(node_set):
    nodes = list(node_set)
    nodes.sort()

    node_id_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = i

    return node_id_map


def load_test_train_data(file, node_id_map, prev_event_dict=None):
    # File can be both the file path or an ordered event_list
    if isinstance(file, str):
        # load the core dataset. sender_id,receiver_id,unix_timestamp
        data = np.loadtxt(file, np.float, delimiter=',', usecols=(0, 1, 2))
        # Sorting by unix_timestamp
        data = data[data[:, 2].argsort()]
    else:
        data = file

    duration = data[-1, 2] - data[0, 2]

    event_dict = {} if prev_event_dict is None else prev_event_dict

    for i in range(data.shape[0]):
        # This step is needed to skip events involving nodes that were not in train, in case they were removed.
        if np.int(data[i, 0]) not in node_id_map or np.int(data[i, 1]) not in node_id_map:
            continue

        sender_id = node_id_map[np.int(data[i, 0])]
        receiver_id = node_id_map[np.int(data[i, 1])]

        if (sender_id, receiver_id) not in event_dict:
            event_dict[(sender_id, receiver_id)] = []

        event_dict[(sender_id, receiver_id)].append(data[i, 2])

    return event_dict, duration


def load_test_train_combined(train, test, node_id_map):
    combined_event_dict, _ = load_test_train_data(train, node_id_map)
    combined_event_dict, _ = load_test_train_data(test, node_id_map, combined_event_dict)

    return combined_event_dict


def split_event_list_to_train_test(event_list, train_percentage=0.8, remove_nodes_not_in_train=False):
    """
    Given an event_list (list of [sender_id, receiver_id, timestamp]) it splits it into train and test,
    ready for model fitting.
    """
    # sort by timestamp
    event_list = event_list[event_list[:, 2].argsort()]
    # make the dataset to start from time 0
    event_list[:, 2] = event_list[:, 2] - event_list[0, 2]

    combined_duration = event_list[-1, 2] - event_list[0, 2]

    split_point = np.int(event_list.shape[0] * train_percentage)

    # Train data
    train_event_list = event_list[:split_point, :]
    train_nodes_set = set(train_event_list[:, 0]).union(train_event_list[:, 1])
    train_node_id_map = get_node_map(train_nodes_set)

    # Test data
    test_event_list = event_list[split_point:, :]
    test_nodes_set = set(test_event_list[:, 0]).union(test_event_list[:, 1])
    if remove_nodes_not_in_train:
        test_nodes_set = test_nodes_set - test_nodes_set.difference(train_nodes_set)
    test_node_id_map = get_node_map(test_nodes_set)

    # Combined
    if remove_nodes_not_in_train:
        combined_node_id_map = train_node_id_map
    else:
        all_nodes = list(train_nodes_set.union(test_nodes_set))
        combined_node_id_map = get_node_map(all_nodes)
        all_nodes.sort()

    nodes_not_in_train = []
    for n in test_nodes_set.difference(train_nodes_set):
        nodes_not_in_train.append(combined_node_id_map[n])

    train_event_dict, train_duration = load_test_train_data(train_event_list, train_node_id_map)
    test_event_dict, test_duration = load_test_train_data(test_event_list, test_node_id_map)
    combined_event_dict = load_test_train_combined(train_event_list, test_event_list, combined_node_id_map)

    return ((train_event_dict, len(train_node_id_map), train_duration),
            (test_event_dict, len(test_node_id_map), test_duration),
            (combined_event_dict, len(combined_node_id_map), combined_duration),
            nodes_not_in_train)


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


def load_facebook_wall(timestamp_max=1000, largest_connected_component_only=False, train_percentage=None):
    """
    :param timestamp_max:
    :param largest_connected_component_only:
    :param train_percentage: If None, returns the entire dataset as a single dataset, else returns a train/test/combined
                             dataset based on the train_percentage.
    """

    file_path = '/shared/DataSets/FacebookViswanath2009/raw/facebook-wall.txt'

    # receiver_id sender_id unix_timestamp
    data = np.loadtxt(file_path, np.float)

    # remove self-edges
    data = data[np.where(data[:, 0] != data[:, 1])[0], :]

    if largest_connected_component_only:
        # finding the nodes in the largest connected component
        fb_net = nx.Graph()
        for i in range(data.shape[0]):
            fb_net.add_edge(data[i, 1], data[i, 0])

        largest_cc = max(nx.connected_components(fb_net), key=len)
        edge_idx_in_largest_cc = np.array([node_id in largest_cc for node_id in data[:, 0]])
        data = data[edge_idx_in_largest_cc, :]

    # Sorting by unix_timestamp and adjusting first timestamp to start from 0
    data = data[data[:, 2].argsort()]
    data[:, 2] = data[:, 2] - data[0, 2]

    if timestamp_max is not None:
        # Scale timestamps to 0 to timestamp_max
        data[:, 2] = (data[:, 2] - min(data[:, 2])) / (max(data[:, 2]) - min(data[:, 2])) * timestamp_max

    if train_percentage is not None:
        return split_event_list_to_train_test(data, train_percentage=train_percentage)
    
    duration = data[-1, 2]

    node_set = set(data[:, 0].astype(np.int)).union(data[:, 1].astype(np.int))
    node_id_map = get_node_map(node_set)

    event_dict = {}
    for i in range(data.shape[0]):
        receiver_id = node_id_map[np.int(data[i, 0])]
        sender_id = node_id_map[np.int(data[i, 1])]

        if (sender_id, receiver_id) not in event_dict:
            event_dict[(sender_id, receiver_id)] = []

        event_dict[(sender_id, receiver_id)].append(data[i, 2])

    return event_dict, len(node_set), duration



if __name__ == '__main__':

    load_facebook_wall(largest_connected_component_only=True)
    # load_core_reality_mining_based_on_dubois()
    # load_reality_mining_test_train()
    # plot_event_count_hist(reality_mining_event_dict, num_nodes, "Reality Mining's Core people")

    # enron_event_dict, num_nodes = load_enron()
    # plot_event_count_hist(enron_event_dict, num_nodes, "Enron")
    # print(load_enron())

    # load_reality_mining_test_train()
    # load_core_reality_mining()

    # ((enron_train_event_dict, enron_train_n_nodes),
    #  (enron_test_event_dict, enron_test_n_nodes),
    #  (enron_combined_event_dict, enron_combined_n_nodes), nodes_not_in_train) = load_enron_train_test()
    #
    # print(np.sum(utils.event_dict_to_aggregated_adjacency(enron_train_n_nodes, enron_train_event_dict)))
    # print(np.sum(utils.event_dict_to_aggregated_adjacency(enron_test_n_nodes, enron_test_event_dict)))
    # print(np.sum(utils.event_dict_to_aggregated_adjacency(enron_combined_n_nodes, enron_combined_event_dict)))

    # ((enron_train_event_dict, enron_train_n_nodes, train_duration),
    #  (enron_test_event_dict, enron_test_n_nodes, test_duration),
    #  (enron_combined_event_dict, enron_combined_n_nodes, combined_duration), nodes_not_in_train) = load_core_reality_mining_based_on_dubois()
    # 
    # print("Train -- Num Nodes:", enron_train_n_nodes,
    #       "Num Edges:", np.sum(utils.event_dict_to_aggregated_adjacency(enron_train_n_nodes, enron_train_event_dict)))
    # print("Test -- Num Nodes:", enron_test_n_nodes,
    #       "Num Edges:", np.sum(utils.event_dict_to_aggregated_adjacency(enron_test_n_nodes, enron_test_event_dict)))
    # print("Combined -- Num Nodes:", enron_combined_n_nodes,
    #       "Num Edges:", np.sum(utils.event_dict_to_aggregated_adjacency(enron_combined_n_nodes, enron_combined_event_dict)))

