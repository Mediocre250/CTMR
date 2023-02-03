import pickle
import numpy as np
import torch
import random
import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt
from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix
import scipy.sparse as sp


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import torch.nn as nn


seed = 0
global_seed = 0
torch.manual_seed(seed)


def load_graph_data(pkl_filename):
    graph = load_pickle(pkl_filename)  # list
    # graph = np.array(graph[0])
    return graph


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ': ', e)
        raise
    return pickle_data


def calculate_preference_similarity(m1, m2, pref):
    """
        m1: (user_len, hidden_size)
        m2: (user_len, seq_len, hidden_size)
        return: calculate the similarity between user and location, which means user's preference about location
    """
    user_len = m1.shape[0]
    seq_len = m2.shape[1]
    pref = pref.squeeze()  # (1, hidden_size) -> (hidden_size, )
    similarity = torch.zeros(user_len, seq_len, dtype=torch.float32)
    for i in range(user_len):
        v1 = m1[i]
        for j in range(seq_len):
            v2 = m2[i][j]
            similarity[i][j] = (1 + torch.cosine_similarity(v1 + pref, v2, dim=0).item()) / 2  # 归一化到[0, 1]

    return similarity


def compute_preference(m1, m2, pref):
    m1 = (m1 + pref).unsqueeze(1)
    s = m1 - m2
    sim = torch.exp(-(torch.norm(s, p=2, dim=-1)))
    return sim


# def calculate_friendship_similarity(u1, m2, pref, device):
#     """
#         u1: (1, hidden_size)
#         m2：(user_count - 1, hidden_size)
#         cur_u: 代表当前用户u1的实际索引id
#         return: calculate the similarity between users, which means user's similarity
#     """
#     user_len = m2.shape[0]
#     pref = pref.squeeze()
#     u1 = u1.squeeze()
#     similarity = torch.zeros(user_len, dtype=torch.float32).to(device)  # (user_count - 1, )
#     for u in range(user_len):
#         u2 = m2[u]
#         similarity[u] = (1 + torch.cosine_similarity(u1 + friend, u2, dim=0).item()) / 2  # 归一化到[0, 1]
#
#     return similarity


def get_user_static_preference(pref, locs):
    """
        pref: (user_len, seq_len)
        locs: (user_len, seq_len, hidden_size)
        return: 返回用户对于所访问POI的全局偏好
    """
    # pref = torch.softmax(pref, dim=1)  # (user_len, seq_len)
    # pref = pref.unsqueeze(2)  # (user_len, seq_len, 1)
    # user_preference = pref * locs  # (user_len, seq_len, hidden_size)
    # user_preference = user_preference.permute(1, 0, 2)  # (seq_len, user_len, hidden_size)
    user_len, seq_len = pref.shape[0], pref.shape[1]
    hidden_size = locs.shape[2]
    user_preference = torch.zeros(user_len, seq_len, hidden_size)
    for i in range(user_len):
        for j in range(seq_len):  # (hidden_size, )
            user_preference[i][j] = torch.sum(torch.softmax(pref[i, :j + 1], dim=0).unsqueeze(1) * locs[i, :j + 1],
                                              dim=0)
    user_preference = user_preference.permute(1, 0, 2)  # (seq_len, user_len, hidden_size)

    return user_preference


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]  # prob (batch_size, loc_count)
    init_label = torch.zeros(num_label, dtype=torch.int64)  # (batch_size, )
    init_prob = torch.zeros(size=(num_label, num_neg + 1))  # (batch_size, num_neg + 1)

    for batch in range(num_label):
        random_ig = random.sample(range(l_m), num_neg)  # (num_neg) from (0 -- l_max - 1)
        while label[batch].item() in random_ig:  # no intersection
            # print('循环查找')
            random_ig = random.sample(range(l_m), num_neg)

        # place the pos labels ahead and neg samples in the end
        for i in range(num_neg + 1):
            if i < 1:
                init_prob[batch, i] = prob[batch, label[batch]]
            else:
                init_prob[batch, i] = prob[batch, random_ig[i - 1]]

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (batch_size, num_neg+1), (batch_size)


def bprLoss(pos, neg, target=1.0):
    loss = - F.logsigmoid(target * (pos - neg))
    return loss.mean()


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def top_transition_graph(transition_graph):
    graph = coo_matrix(transition_graph)
    data = graph.data
    row = graph.row
    threshold = 20

    for i in range(0, row.size, threshold):
        row_data = data[i: i + threshold]
        norm = row_data.max()
        row_data = row_data / norm
        data[i: i + threshold] = row_data

    return graph


def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return graph


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx  # D^-1 W


def calculate_reverse_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


if __name__ == '__main__':
    graph_path = 'data/user_similarity_graph.pkl'
    user_similarity_matrix = torch.tensor(load_graph_data(pkl_filename=graph_path))
    print(user_similarity_matrix[1])
    print('................')
    print(user_similarity_matrix[1][:10])
    count = 0
    # for i in range(user_similarity_matrix.shape[0]):
    #     for j in range(user_similarity_matrix.shape[1]):
    #         if user_similarity_matrix[i][j] > 0.01:  # 5747013, 即9.5%
    #             count += 1

    print('count: ', count)















def propertyFunc_var(adj_matrix):
    return adj_matrix.var()

def propertyFunc_mean(adj_matrix):
    return adj_matrix.mean()

def propertyFunc_std(adj_matrix):
    return adj_matrix.std()

def propertyFunc_UnidirectionalIndex(adj_matrix):
    unidirectionalIndex = 0
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[0])):
            unidirectionalIndex = unidirectionalIndex +\
                                  abs(adj_matrix[i][j] - adj_matrix[j][i])
    return unidirectionalIndex

def getPropertyArrayWithPropertyFunc(data_input, property_func):
    result = []
    for i in range(len(data_input)):
        result.append(property_func(data_input[i]))
    # -- standardlize
    return np.array(result)

def getDistanceMatrixWithPropertyArray(data_x, property_array, isSigmoid=False):
    sampleNum = data_x.shape[0]
    disMatrix = np.zeros([sampleNum, sampleNum])
    for i in range(0, sampleNum):
        for j in range(0, sampleNum):
            if isSigmoid:
                hour_i = i % 24
                hour_j = j % 24
                hour_dis = abs(hour_i-hour_j)
                if hour_dis == 23:
                    hour_dis = 1
                c = sigmoid(hour_dis/24)
            else:
                c = 1
            disMatrix[i][j] = c * abs(property_array[i] - property_array[j])
    disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
    return disMatrix

def getDistanceMatrixWithPropertyFunc(data_x, property_func, isSigmoid=False):
    property_array = getPropertyArrayWithPropertyFunc(data_x, property_func)
    disMatrix = getDistanceMatrixWithPropertyArray(data_x, property_array, isSigmoid=isSigmoid)
    return disMatrix

def get_SSEncode2D(one_data, mean_data):
    result = []
    for i in range(len(one_data)):
        for j in range(len(one_data[0])):
            if one_data[i][j] > mean_data[i][j]:
                result.append(1)
            else:
                result.append(0)
    return np.array(result)

def getDistanceMatrixWith_SSIndex(input_data, isSigmoid=True):
    sampleNum = len(input_data)
    input_data_mean = input_data.mean(axis=0)
    property_array = []
    for i in range(len(input_data)):
        property_array.append(get_SSEncode2D(input_data[i], input_data_mean))
    property_array = np.array(property_array)
    disMatrix = np.zeros([sampleNum, sampleNum])
    for i in range(0, sampleNum):
        for j in range(0, sampleNum):
            if isSigmoid:
                hour_i = i % 24
                hour_j = j % 24
                sub_hour = abs(hour_i-hour_j)
                if sub_hour == 23:
                    sub_hour = 1
                c = sigmoid(sub_hour/24)
            else:
                c = 1
            sub_encode = abs(property_array[i] - property_array[j])
            disMatrix[i][j] = c * sub_encode.sum()
    disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
    label_pred = getClusterLabelWithDisMatrix(disMatrix, display_dis_matrix=False)
    return disMatrix

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Mobility_Graph_Distance(m_graphs):
    """
    :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
    :return: (N, N). Distance matrix between every two graphs
    """
    # Mean
    isSigmoid = True
    mean_dis_matrix = getDistanceMatrixWithPropertyFunc(
        m_graphs, propertyFunc_mean, isSigmoid=isSigmoid)
    # Uniflow
    unidirIndex_dis_matrix = getDistanceMatrixWithPropertyFunc(
        m_graphs, propertyFunc_UnidirectionalIndex, isSigmoid=isSigmoid
    )
    # Var
    var_dis_matrix = getDistanceMatrixWithPropertyFunc(
        m_graphs, propertyFunc_var, isSigmoid=isSigmoid
    )
    # SS distance
    ss_dis_matrix = getDistanceMatrixWith_SSIndex(m_graphs, isSigmoid=isSigmoid)
    c_mean_dis = 1
    c_unidirIndex_dis = 1
    c_std_dis = 1
    c_ss_dis = 1
    disMatrix = (c_mean_dis * mean_dis_matrix) \
                + (c_unidirIndex_dis * unidirIndex_dis_matrix) \
                + (c_std_dis * var_dis_matrix) \
                + (c_ss_dis * ss_dis_matrix)
    return disMatrix

def getClusterLabelWithDisMatrix(dis_matrix, display_dis_matrix=False):
    n_clusters = 7
    # # linkage: single, average, complete
    linkage = "complete"
    # ---
    # t1 = time.time()
    if display_dis_matrix:
        sns.heatmap(dis_matrix)
        plt.show()
    # ---
    estimator = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
    estimator.fit(dis_matrix)
    label_pred = estimator.labels_
    # print("The time consuming of clustering (known disMatrix)：", time.time() - t1)
    return label_pred

def getPatternWithMGD(m_graphs):
    """
    :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
    :return mob_patterns: (n_clusters, M, M). n_clusters patterns, each pattern contains a graph
    :return cluster_label: 
    """
    n_clusters = 7
    linkage = "complete"
    disMatrix = Mobility_Graph_Distance(m_graphs)
    # -- Agglomerative Cluster
    estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
    estimator.fit(disMatrix)
    label_pred = estimator.labels_
    # cluster_label = label_pred.reshape((31, 24))
    # -- Generate Mobility Pattern
    patterns = []
    for i in range(n_clusters):
        this_cluster_index_s = np.argwhere(label_pred == i).flatten()
        this_cluster_graph_s = m_graphs[this_cluster_index_s]
        patterns.append(this_cluster_graph_s.sum(axis=0))
    mob_patterns = np.array(patterns)
    return mob_patterns, label_pred









class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 输入层，隐藏层*2,输出层.隐藏层节点数目为输入层两倍
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )

        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output


class ConcatLinear(nn.Module):
    """
    input: (*, a) and (*, b)
    output (*, c)
    """
    def __init__(self, in_1, in_2, out, dropout=0.1):
        super(ConcatLinear, self).__init__()
        self.linear1 = nn.Linear(in_1+in_2, out)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out)

        self.linear2 = nn.Linear(out, out)
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x1, x2):
        src = torch.cat([x1, x2], -1)
        out = self.linear1(src)
        out = src + self.dropout1(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.act2(self.linear2(out))
        return out


class GraphStructuralEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,):
        super(GraphStructuralEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):
        src2 = self.self_attn(src, src, src,)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src






def pairwise_inner_product(mat_1, mat_2):
    n, m = mat_1.shape  # (180, 144)
    mat_expand = torch.unsqueeze(mat_2, 0)  # (1, 180, 144),
    mat_expand = mat_expand.expand(n, n, m)  # (180, 180, 144),
    mat_expand = mat_expand.permute(1, 0, 2)  # (180, 180, 144),
    inner_prod = torch.mul(mat_expand, mat_1)  # (180, 180, 144), 
    inner_prod = torch.sum(inner_prod, axis=-1)  # (180, 180),
    return inner_prod


def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = pairwise_inner_product(s_embeddings, t_embeddings)
    softmax1 = nn.Softmax(dim=-1)
    phat = softmax1(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat+0.0001)))
    inner_prod = pairwise_inner_product(t_embeddings, s_embeddings)
    softmax2 = nn.Softmax(dim=-1)
    phat = softmax2(inner_prod)
    loss += torch.sum(-torch.mul(torch.transpose(mob, 0, 1), torch.log(phat+0.0001)))
    return loss


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, out1, out2, label):

        mob_loss = _mob_loss(out1, out2, label)
        loss = mob_loss
        return loss