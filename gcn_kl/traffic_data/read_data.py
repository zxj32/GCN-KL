import json as json, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import random
from scipy import sparse
from collections import Counter

edge_path = "/network/rit/home/xz381633/traffic_dataset/Philly_Data/philly_graph_edge.txt"
node_path = "/network/rit/home/xz381633/traffic_dataset/Philly_Data/philly_graph_node.txt"
feature_path = "/network/rit/home/xz381633/traffic_dataset/DC_Data/hourly_data/hour_10_weekday_0_speed.json"
feature_file = "/network/rit/home/xz381633/traffic_dataset/Philly_Data/hourly_data/"
feature_path_ = "hour_14_weekday_2_speed.json"


def get_feature(f_path, traffic_speed):
    edge_file = np.load("/network/rit/home/xz381633/traffic_deep/input_data/edge.npz")
    edge_tid = edge_file["arr_1"]
    feature = np.zeros(len(edge_tid))
    feature_traffic = np.zeros(len(edge_tid))
    f_path = "/network/rit/home/xz381633/traffic_dataset/DC_Data/hourly_data/" + f_path
    with open(f_path) as feat:
        for line in feat:
            f_line = eval(line)
            data = f_line['data']
            data = filter(lambda a: a != [], data)
            if data == []:
                data = [[[0.0, 0.0, 0.0]]]
            data_mean = np.mean(data, axis=0)
            tmc = f_line['tmc']
            id_ = np.where(edge_tid == tmc)
            id_edge = id_[0][0]
            feature[id_edge] = data_mean[0][1]
            if data_mean[0][1] > traffic_speed:
                feature_traffic[id_edge] = -1.0  # non-congestion
            else:
                feature_traffic[id_edge] = 1.0  # congestion

    return feature, feature_traffic


def get_features(f_path):
    edge_file = np.load("/network/rit/home/xz381633/traffic_deep/input_data/edge_ph.npz")
    edge_tid = edge_file["arr_1"]
    f_path = "/network/rit/home/xz381633/traffic_dataset/Philly_Data/hourly_data/" + f_path
    features = np.zeros(shape=[len(edge_tid), 44])
    with open(f_path) as feat:
        for line in feat:
            f_line = eval(line)
            tmc = f_line['tmc']
            id_ = np.where(edge_tid == tmc)
            id_edge = id_[0][0]
            data = f_line['data']
            data = filter(lambda a: a != [], data)
            if data:
                for i in range(len(data)):
                    speed = data[i]
                    if speed:
                        if speed[0][0] < 25.0:
                            features[id_edge][i] = 1.0  # congestion
                        else:
                            features[id_edge][i] = -1.0  # non-congestion
                    else:
                        features[id_edge][i] = 0.0  # non-congestion
    return features, features


def get_node_index(u, v):
    u_index = -1
    v_index = -1
    with open(node_path) as node:
        for n in node:
            n = eval(n)
            if n['node'] == u:
                u_index = n['index']
            if n['node'] == v:
                v_index = n['index']
    return u_index, v_index


def get_edge():
    edge = []
    edge_id = []
    with open(edge_path) as f:
        content = f.readlines()
        for line in content:
            line = eval(line)
            u, v = get_node_index(line['u'], line['v'])
            edge.append(u)
            edge.append(v)
            edge_id.append(line['tmc']['tmc_id'])
    return edge, edge_id


def get_i_neighbors(ui, vi, edge, i):
    u_neigh = []
    v_neigh = []
    for k in range(len(edge)):
        edge_k = edge[k]
        if edge_k[0] == ui:
            u_neigh.append(k)
        if edge_k[1] == ui:
            u_neigh.append(k)
        if edge_k[0] == vi:
            v_neigh.append(k)
        if edge_k[1] == vi:
            v_neigh.append(k)
    u_neigh.remove(i)
    v_neigh.remove(i)
    return u_neigh, v_neigh


def get_neighbor():
    edge_file = np.load("/network/rit/home/xz381633/traffic_deep/input_data/edge_ph.npz")
    edge = edge_file["arr_0"]
    edge = np.reshape(edge, [len(edge)/2, 2])
    edge_tid = edge_file["arr_1"]
    u_neighbor = []
    v_neighbor = []
    for i in range(len(edge_tid)):
        ui = edge[i][0]
        vi = edge[i][1]
        ui_neigh, vi_neigh = get_i_neighbors(ui, vi, edge, i)
        u_neighbor.append(ui_neigh)
        v_neighbor.append(vi_neigh)
    np.savez("/network/rit/home/xz381633/traffic_deep/input_data/edge_neighbor_ph.npz", u_neighbor, v_neighbor)
    return u_neighbor, v_neighbor


def get_cluster():
    edge_file = np.load("edge.npz")
    edge = edge_file["arr_0"]
    edge_tid = edge_file["arr_1"]
    edge_neigh_length = []
    edge_single = []
    edge_neighbor = np.load("edge_neighbor.npz")
    u_neigh = edge_neighbor["arr_0"]
    v_neigh = edge_neighbor["arr_1"]
    for i in range(len(u_neigh)):
        edge_neigh_length.append([len(u_neigh[i]), len(v_neigh[i])])
        if len(u_neigh[i]) == 0:
            edge_single.append([len(u_neigh[i]), len(v_neigh[i]), i])
        elif len(v_neigh[i]) == 0:
            edge_single.append([len(u_neigh[i]), len(v_neigh[i]), i])

    edge_neigh_length_all = np.sum(np.array(edge_neigh_length), axis=1)
    sort_ = np.sort(edge_neigh_length_all)
    # cluster
    clf = KMeans(n_clusters=3, random_state=0)
    s = clf.fit(edge_neigh_length)
    numSamples = len(edge_neigh_length)
    centroids = clf.labels_
    return centroids


def save_all_feature():
    feature_edge = []
    for filename in os.listdir(feature_file):
        feature_edgei, _ = get_features(filename)
        feature_edge.append(feature_edgei)
        # print(Counter(feature_edgei))
        # for i in range(35):
        #     tt = feature_edgei[:, i]
        #     c = Counter(tt).values()
        #     if c[1] < 522:
        #         print(c)
        #     elif c[2] > 1000:
        #         print(c)
    np.save("/network/rit/home/xz381633/traffic_deep/input_data/features_all_25_ph.npy", feature_edge)
    return


def get_neighbor_feat(which_feat):
    feature_edge = np.load("feature/feature_edge_25.npy")
    edge_file = np.load("edge.npz")
    edge = edge_file["arr_0"]
    edge_neighbor = np.load("edge_neighbor.npz")
    u_neigh = edge_neighbor["arr_0"]
    v_neigh = edge_neighbor["arr_1"]

    feature_neighbor_u = []
    feature_neighbor_v = []
    feature_edge_0 = feature_edge[which_feat]
    for i in range(len(u_neigh)):
        fu = []
        fv = []
        for edgeu in u_neigh[i]:
            fu.append(feature_edge_0[edgeu])
        for edgev in v_neigh[i]:
            fv.append(feature_edge_0[edgev])
        feature_neighbor_u.append(fu)
        feature_neighbor_v.append(fv)
    return feature_neighbor_u, feature_neighbor_v


def get_adjacency_matrix():
    edge_neighbor = np.load("/network/rit/home/xz381633/traffic_deep/input_data/edge_neighbor_ph.npz")
    u_neigh = edge_neighbor["arr_0"]
    v_neigh = edge_neighbor["arr_1"]
    adjacency_matrix = np.zeros([len(u_neigh), len(u_neigh)])
    for i in range(len(u_neigh)):
        neigh = np.hstack((u_neigh[i], v_neigh[i]))
        for j in neigh:
            j = int(j)
            adjacency_matrix[i][j] = int(1)
    return adjacency_matrix


def get_degree_matrix(ad_matrix):
    a, b = np.shape(ad_matrix)
    degree_matrix = np.zeros_like(ad_matrix)
    ad_sum = np.sum(ad_matrix, axis=1)
    for i in range(a):
        degree_matrix[i][i] = ad_sum[i]
    return degree_matrix


def get_laplacian_matrix(adjacency):
    degree = get_degree_matrix(adjacency)
    laplacian_matrix = degree - adjacency
    return laplacian_matrix


def adjacency_matrix_normalized(adjacency):
    a, _ = np.shape(adjacency)
    ad_matrix = adjacency + np.diag(np.ones(a))
    deg_matrix = get_degree_matrix(ad_matrix)


def get_hotvalue(feat):
    hotvalue = np.zeros([len(feat), 2])
    for i in range(len(feat)):
        if feat[i] == -1.0:  # non-conjestion
            # feat[i] = 100.0
            hotvalue[i] = [0, 1]
        elif feat[i] == 1.0:  # conjestion
            hotvalue[i] = [1, 0]
        else:  # unknown node
            # feat[i] = 50.0
            hotvalue[i] = [0, 0]
    return hotvalue, feat


def get_hotvalue_f(feat):
    hotvalue = np.zeros([len(feat), 2])
    feat_n = np.zeros([len(feat), 3])
    for i in range(len(feat)):
        if feat[i] == -1.0:  # non-conjestion
            feat_n[i] = [0, 0, 1]
            hotvalue[i] = [0, 1]
        elif feat[i] == 1.0:  # conjestion
            feat_n[i] = [1, 0, 0]
            hotvalue[i] = [1, 0]
        else:  # unknown node
            feat_n[i] = [0, 1, 0]
            hotvalue[i] = [0, 0]
    return hotvalue, feat_n


def get_hotvalue_bigdata(feat):
    hotvalue = np.zeros([len(feat), 2])
    feat_n = np.zeros([len(feat), 3])
    for i in range(len(feat)):
        if feat[i] == 0:  # non-conjestion
            feat_n[i] = [0, 0, 1]
            hotvalue[i] = [0, 1]
        elif feat[i] == 1:  # conjestion
            feat_n[i] = [1, 0, 0]
            hotvalue[i] = [1, 0]
        else:  # unknown node
            feat_n[i] = [0, 1, 0]
            hotvalue[i] = [0, 0]
    return hotvalue, feat_n


def load_data(week, test_num, hour):
    random.seed(132)
    adj_n = get_adjacency_matrix()
    features_ = np.load("/network/rit/home/xz381633/traffic_deep/input_data/features_hour14_week2_25.npy")
    # feature_hour = features_[hour]
    feature_edge_i = features_[:, week]
    test_index = random.sample(range(len(feature_edge_i)), test_num)
    label, feature_n = get_hotvalue_f(feature_edge_i)

    # feature_n = feature_edge_i
    y_train = np.zeros_like(label)
    y_test = np.zeros_like(label)
    train_mask = np.zeros_like(feature_edge_i, dtype=bool)
    test_mask = np.zeros_like(feature_edge_i, dtype=bool)
    for index in test_index:
        feature_n[index] = [0, 1, 0]
    for i in range(len(test_mask)):
        if i in test_index:
            y_test[i] = label[i]
            test_mask[i] = True
        else:
            y_train[i] = label[i]
            train_mask[i] = True
    y_val = y_test
    val_mask = test_mask
    adj = sparse.csr_matrix(adj_n)
    # features = sparse.csr_matrix(np.reshape(feature_n, [len(feature_n), 1]))
    features = sparse.csr_matrix(feature_n)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_hour(week, test_num, hour):
    random.seed(132)
    # adj_n = get_adjacency_matrix()
    adj_n = np.load("/network/rit/home/xz381633/traffic_deep/input_data/adjacency_matrix_ph.npy")
    features_ = np.load("/network/rit/home/xz381633/traffic_deep/input_data/features_all_25_ph.npy")
    feature_hour = features_[hour]
    feature_edge_i = feature_hour[:, week]
    test_index = random.sample(range(len(feature_edge_i)), test_num)
    label, feature_n = get_hotvalue_f(feature_edge_i)

    # feature_n = feature_edge_i
    y_train = np.zeros_like(label)
    y_test = np.zeros_like(label)
    train_mask = np.zeros_like(feature_edge_i, dtype=bool)
    test_mask = np.zeros_like(feature_edge_i, dtype=bool)
    for index in test_index:
        feature_n[index] = [0, 1, 0]
    for i in range(len(test_mask)):
        if i in test_index:
            y_test[i] = label[i]
            test_mask[i] = True
        else:
            y_train[i] = label[i]
            train_mask[i] = True
    y_val = y_test
    val_mask = test_mask
    adj = sparse.csr_matrix(adj_n)
    # features = sparse.csr_matrix(np.reshape(feature_n, [len(feature_n), 1]))
    features = sparse.csr_matrix(feature_n)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_traffic(week, test_num, hour):
    random.seed(132)
    # adj_n = get_adjacency_matrix()
    adj_n = np.load("./data/adjacency_matrix_dc_milcom.npy")
    features_ = np.load("./data/feature_dc_0.8_6hours_Fr.npy")
    feature_hour = features_[hour]
    feature_edge_i = feature_hour[:, week]
    test_index = random.sample(range(len(feature_edge_i)), test_num)
    label, feature_n = get_hotvalue_bigdata(feature_edge_i)

    # feature_n = feature_edge_i
    y_train = np.zeros_like(label)
    y_test = np.zeros_like(label)
    train_mask = np.zeros_like(feature_edge_i, dtype=bool)
    test_mask = np.zeros_like(feature_edge_i, dtype=bool)
    for index in test_index:
        feature_n[index] = [0, 1, 0]
    for i in range(len(test_mask)):
        if i in test_index:
            y_test[i] = label[i]
            test_mask[i] = True
        else:
            y_train[i] = label[i]
            train_mask[i] = True
    y_val = y_test
    val_mask = test_mask
    adj = sparse.csr_matrix(adj_n)
    # features = sparse.csr_matrix(np.reshape(feature_n, [len(feature_n), 1]))
    features = sparse.csr_matrix(feature_n)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def feature_random(feature, index):
    k = 0
    num = 100
    # random.seed()
    # print(Counter(feature))
    r_index = random.sample(range(len(feature)), len(index) + num)
    for item in r_index:
        if item in index:
            pass
        else:
            feature[item] = [0, 1, 0]
            k = k + 1
        if k == num:
            break
    # print(Counter(feature))
    return feature


def get_opinion(prediction, test_mask):
    prod = np.prod(prediction, axis=0)
    prod_pow = np.power(prod, 1.0 / len(prediction))
    opinion = np.zeros_like(prod_pow)
    opinion_label = np.zeros_like(prod_pow)
    for i in range(len(opinion)):
        if test_mask[i]:
            opinion[i][0] = prod_pow[i][0] / (prod_pow[i][0] + prod_pow[i][1])
            opinion[i][1] = prod_pow[i][1] / (prod_pow[i][0] + prod_pow[i][1])
            if prod_pow[i][0] > prod_pow[i][1]:
                opinion_label[i][0] = 1.0
                opinion_label[i][1] = 0.0
            elif prod_pow[i][0] < prod_pow[i][1]:
                opinion_label[i][0] = 0.0
                opinion_label[i][1] = 1.0
    return opinion, opinion_label


def opinion_c(y_list):
    weeks = float(len(y_list))
    y_round = np.around(y_list)
    opinion = np.sum(y_round, axis=0) / weeks
    return opinion


def opinion_error(opinion1, opinion2, mask):
    error = opinion1 - opinion2
    error = np.abs(error[:, 0])
    mask_ = np.asarray(mask, dtype=float)
    mask_ /= np.mean(mask_)
    error = error * mask_
    error_o = np.mean(error)
    return error_o



