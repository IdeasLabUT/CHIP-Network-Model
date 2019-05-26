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

    return event_dict, len(people)


def load_enron_train_test():
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

    return load_train_test(train_file_path, test_file_path, combined_duration)


def load_fb_train_test():
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

    return load_train_test(train_file_path, test_file_path, combined_duration)


def load_train_test(train_file_path, test_file_path, combined_duration):
    """
    Loads datasets already split into train and test, such as Enron and FB.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """

    combined_node_id_map, train_node_id_map, test_node_id_map, nodes_not_in_train = \
        load_and_combine_nodes_for_test_train(train_file_path, test_file_path)

    train_event_dict, train_duration = load_test_train_data(train_file_path, train_node_id_map)
    test_event_dict, test_duration = load_test_train_data(test_file_path, test_node_id_map)
    combined_event_dict = load_test_train_combined(train_file_path, test_file_path, combined_node_id_map)

    return ((train_event_dict, len(train_node_id_map), train_duration),
            (test_event_dict, len(test_node_id_map), test_duration),
            (combined_event_dict, len(combined_node_id_map), combined_duration),
            nodes_not_in_train)


def load_and_combine_nodes_for_test_train(train_path, test_path):
    """
    Loads the set of nodes in both train and test datasets and maps all the node ids to start form 0 to num total nodes.
    :param train_path
    :param test_path
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
    test_node_id_map = get_node_map(test_nodes_set)

    # Combined
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


def split_event_list_to_train_test(event_list, train_percentage=0.8):
    """
    Given an event_list it splits it into train and test, ready for model fitting.
    """
    # sort by timestamp
    event_list = event_list[event_list[:, 2].argsort()]
    combined_duration = event_list[-1, 2] - event_list[0, 2]

    split_point = np.int(event_list.shape[0] * train_percentage)

    # Train data
    train_event_list = event_list[:split_point, :]
    train_nodes_set = set(train_event_list[:, 0]).union(train_event_list[:, 1])
    train_node_id_map = get_node_map(train_nodes_set)

    # Test data
    test_event_list = event_list[split_point:, :]
    test_nodes_set = set(test_event_list[:, 0]).union(test_event_list[:, 1])
    test_node_id_map = get_node_map(test_nodes_set)

    # Combined
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


# def load_facebook_wall():
#     jan_1_2007 = 1167609600
#     jan_1_2008 = 1199145600
#     file_path = '/shared/DataSets/FacebookViswanath2009/raw/facebook-wall.txt'
#
#     # load the core dataset. sender_id  receiver_id unix_timestamp
#     data = np.loadtxt(file_path, np.int)
#     # Sorting by unix_timestamp
#     data = data[data[:, 2].argsort()]
#
#     # filtering date
#     data = data[np.where(np.logical_and(data[:, 2] >= jan_1_2007, data[:, 2] <= jan_1_2008))[0], :]
#
#     # remove self-edges
#     data = data[np.where(data[:, 0] != data[:, 1])[0], :]
#
#     # filtering nodes
#     sender_nodes, nodes_out_degree = np.unique(data[:, 0], return_counts=True)
#     receiver_nodes, nodes_in_degree = np.unique(data[:, 1], return_counts=True)
#     nodes_to_include = set(sender_nodes[nodes_out_degree > 9]).intersection(receiver_nodes[nodes_in_degree > 9])
#
#     # Adjust first timestamp to start from 0
#     data[:, 2] = data[:, 2] - data[0, 2]
#
#     # Spilt into train and test
#     num_test_events = 10000
#     combined_data = data
#     full_node_id_map, _ = get_facebook_node_map(data)
#
#     train_data = data[:-num_test_events, :]
#     train_node_id_map, train_nodes_set = get_facebook_node_map(train_data)
#
#     test_data = data[num_test_events:, :]
#     test_node_id_map, test_nodes_set = get_facebook_node_map(test_data)
#
#     nodes_not_in_train = []
#     for n in test_nodes_set.difference(train_nodes_set):
#         nodes_not_in_train.append(full_node_id_map[n])
#
#
#     print(len(nodes_to_include))
#     # print(node_out_degree)
#     # print(data.shape[0])
#
#
# def load_facebook_wall():
#     jan_1_2007 = 1167609600
#     jan_1_2008 = 1199145600
#     file_path = '/shared/DataSets/FacebookViswanath2009/FB_event_mat.csv'
#
#     # load the core dataset. sender_id  receiver_id unix_timestamp
#     data = np.loadtxt(file_path, np.float, delimiter=',')
#     # Sorting by unix_timestamp
#     # data = data[data[:, 2].argsort()]
#
#     print(data.shape[0] * .8)
#     print(data[0:5, :])
#
#     print(data[-1, :])
#
#
#     exit()
#     np.savetxt("/shared/DataSets/FacebookViswanath2009/train_FB_event_mat.csv", data[:-1000, :],
#                fmt="%d")
#
#     exit()
#     # filtering date
#     data = data[np.where(np.logical_and(data[:, 2] >= jan_1_2007, data[:, 2] <= jan_1_2008))[0], :]
#
#     # remove self-edges
#     data = data[np.where(data[:, 0] != data[:, 1])[0], :]
#
#     # filtering nodes
#     sender_nodes, nodes_out_degree = np.unique(data[:, 0], return_counts=True)
#     receiver_nodes, nodes_in_degree = np.unique(data[:, 1], return_counts=True)
#     nodes_to_include = set(sender_nodes[nodes_out_degree > 9]).intersection(receiver_nodes[nodes_in_degree > 9])
#
#     # Adjust first timestamp to start from 0
#     data[:, 2] = data[:, 2] - data[0, 2]
#
#     # Spilt into train and test
#     num_test_events = 10000
#     combined_data = data
#     full_node_id_map, _ = get_facebook_node_map(data)
#
#     train_data = data[:-num_test_events, :]
#     train_node_id_map, train_nodes_set = get_facebook_node_map(train_data)
#
#     test_data = data[num_test_events:, :]
#     test_node_id_map, test_nodes_set = get_facebook_node_map(test_data)
#
#     nodes_not_in_train = []
#     for n in test_nodes_set.difference(train_nodes_set):
#         nodes_not_in_train.append(full_node_id_map[n])
#
#
#     print(len(nodes_to_include))
#     # print(node_out_degree)
#     # print(data.shape[0])
#
#
# def get_facebook_node_map(data):
#     nodes_set = set(data[:, 0]).union(data[:, 1])
#     nodes = list(nodes_set)
#     nodes.sort()
#
#     node_id_map = {}
#     for i, n in enumerate(nodes):
#         node_id_map[n] = i
#
#     return node_id_map, nodes_set

if __name__ == '__main__':
    # reality_mining_event_dict, num_nodes = load_core_reality_mining()
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

    ((enron_train_event_dict, enron_train_n_nodes, train_duration),
     (enron_test_event_dict, enron_test_n_nodes, test_duration),
     (enron_combined_event_dict, enron_combined_n_nodes, combined_duration), nodes_not_in_train) = load_fb_train_test()

    print("Train -- Num Nodes:", enron_train_n_nodes,
          "Num Edges:", np.sum(utils.event_dict_to_aggregated_adjacency(enron_train_n_nodes, enron_train_event_dict)))
    print("Test -- Num Nodes:", enron_test_n_nodes,
          "Num Edges:", np.sum(utils.event_dict_to_aggregated_adjacency(enron_test_n_nodes, enron_test_event_dict)))
    print("Combined -- Num Nodes:", enron_combined_n_nodes,
          "Num Edges:", np.sum(utils.event_dict_to_aggregated_adjacency(enron_combined_n_nodes, enron_combined_event_dict)))

