import torch
import torch.nn as nn
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size * 2, hidden_size, nonlinearity='tanh')  # 因为拼接了time embedding
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size * 2, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size & 2, hidden_size)

        
class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight,
                 graph, spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, loc2grid, user_region_graph, device):
        super().__init__()
        self.device = device
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph

        self.I = identity(graph.shape[0], format='coo')
        self.graph = sparse_matrix_to_tensor(
            calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))

        self.spatial_graph = spatial_graph
        # self.I_f = identity(friend_graph.shape[0], format='coo')
        # self.friend_graph = calculate_random_walk_matrix(
        #     (self.lambda_user * friend_graph + self.I_f).astype(np.float32)).tocsr()
        if interact_graph is not None:
            self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
                interact_graph))  # (M, N)
        else:
            self.interact_graph = None

        # self.region_num = 224

        self.region_num = 264
        self.encoder = nn.Embedding(
            input_size, hidden_size)  # location embedding
        # self.time_encoder = nn.Embedding(24 * 7, hidden_size)  # time embedding
        self.user_encoder = nn.Embedding(
            user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size) 
        self.fc = nn.Linear(2 * hidden_size, input_size)
        self.fc_region = nn.Linear(2 * hidden_size, self.region_num)

        self.user_region_graph = torch.from_numpy(user_region_graph).to(device)

        # Regional embedding
        # self.taxi_region_model = MGFN(graph_num=7, node_num=self.region_num, output_dim=hidden_size)
        # self.taxi_mob_pattern = torch.Tensor(np.load("./trip_data/taxi_mob_pattern.npy"))
        # self.taxi_mob_adj = torch.Tensor(np.load("./trip_data/taxi_mob_label.npy"))
        
        self.fsq_region_model = MGFN(graph_num=7, node_num=self.region_num, output_dim=hidden_size)
        self.fsq_mob_pattern = torch.Tensor(np.load("./trip_data/fsq_mob_pattern.npy"))
        self.fsq_mob_adj = torch.Tensor(np.load("./trip_data/fsq_mob_label.npy"))

        self.region_fuse = RegionFuse(hidden_size, hidden_size, hidden_size)
        self.loc_region_idx = torch.LongTensor([loc2grid[loc] for loc in range(input_size)])

        # self.region_embedding = nn.Embedding(264, hidden_size)
        # self_region_embedding = torch.from_numpy(np.load('trip_data/fsq_region_embedding.npy'))
        self_region_embedding = nn.Embedding(self.region_num, hidden_size)
        cross_region_embedding = torch.from_numpy(np.load('trip_data/fsq_region_embedding.npy')).to(device)
        cross_region_embedding.requires_grad = False
        self.cross_region_model = CrossRegionModel(hidden_size, self_region_embedding, cross_region_embedding, torch.from_numpy(user_region_graph), device)


    def forward(self, x, t, t_slot, s, r, y_t, y_t_slot, y_s, y_r, h, active_user):
        # 用GCN处理转移graph, 即用顶点i的邻居顶点j来更新i所对应的POI embedding
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)

        # 是否用GCN来更新user embedding
        # if self.use_graph_user:
        #     # I_f = identity(self.friend_graph.shape[0], format='coo')
        #     # friend_graph = (self.friend_graph * self.lambda_user + I_f).astype(np.float32)
        #     # friend_graph = calculate_random_walk_matrix(friend_graph)
        #     # friend_graph = sparse_matrix_to_tensor(friend_graph).to(x.device)
        #     friend_graph = self.friend_graph.to(x.device)
        #     # AX
        #     user_emb = self.user_encoder(torch.LongTensor(
        #         list(range(self.user_count))).to(x.device))
        #     user_encoder_weight = torch.sparse.mm(friend_graph, user_emb).to(
        #         x.device)  # (user_count, hidden_size)

        #     if self.use_weight:
        #         user_encoder_weight = self.user_gconv_weight(
        #             user_encoder_weight)
        #     p_u = torch.index_select(
        #         user_encoder_weight, 0, active_user.squeeze())
        # else:
        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        # (user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)

        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)
        # AX,即GCN
        graph = self.graph.to(x.device)
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(
            x.device)  # (input_size, hidden_size)
        
        # if self.use_spatial_graph:
        #     spatial_graph = (self.spatial_graph *
        #                      self.lambda_loc + self.I).astype(np.float32)
        #     spatial_graph = calculate_random_walk_matrix(spatial_graph)
        #     spatial_graph = sparse_matrix_to_tensor(
        #         spatial_graph).to(x.device)  # sparse tensor gpu
        #     encoder_weight += torch.sparse.mm(spatial_graph,
        #                                       loc_emb).to(x.device)
        #     encoder_weight /= 2  # 求均值
       
        # new_x_emb = []
        # for i in range(seq_len):
        #     # (user_len, hidden_size)
        #     temp_x = torch.index_select(encoder_weight, 0, x[i])
        #     new_x_emb.append(temp_x)

        # x_emb = torch.stack(new_x_emb, dim=0)

        # s_out1, t_out1 = self.taxi_region_model(self.taxi_mob_pattern.to(x.device))
        # s_out2, t_out2 = self.fsq_region_model(self.fsq_mob_pattern.to(x.device))




        new_x_embedding = []
        # new_x_embed = []
        for i in range(seq_len):
            # (user_len, hidden_size)
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            region_preference = self.cross_region_model(active_user.squeeze(), r[i])
            new_x_embedding.append(torch.concat((temp_x, region_preference), dim=1))
            # new_x_embed.append(temp_x)


        new_x_embedding = torch.stack(new_x_embedding, dim=0)
        # new_x_embed = torch.stack(new_x_embed, dim=0)
        

        # user-poi and region
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        region_idx = self.loc_region_idx.to(x.device)



        interact_graph = self.interact_graph.to(self.device)
        encoder_weight_user = torch.sparse.mm(interact_graph, encoder_weight).to(self.device)

        # encoder_weight_region = torch.einsum('ij,jk->ik', self.user_region_graph, self.cross_region_model.self_region_embedding.weight).to(self.device)
        encoder_weight_region = self.cross_region_model.self_region_preference

        preferences = torch.concat([encoder_weight_user, encoder_weight_region], axis=1).to(self.device)

        user_preference = torch.index_select(preferences, 0, active_user.squeeze()).unsqueeze(0)
        # print(user_preference.size())
        user_loc_similarity = torch.exp(-(torch.norm(user_preference - new_x_embedding, p=2, dim=-1))).to(x.device)
        user_loc_similarity = user_loc_similarity.permute(1, 0)

        # encoder_weight = loc_emb
        # interact_graph = self.interact_graph.to(x.device)
        # encoder_weight_user = torch.sparse.mm(interact_graph, encoder_weight).to(x.device)

        # user_preference = torch.index_select(encoder_weight_user, 0, active_user.squeeze()).unsqueeze(0)
        # # print(user_preference.size())
        # user_loc_similarity = torch.exp(-(torch.norm(user_preference - new_x_embed, p=2, dim=-1))).to(x.device)
        # user_loc_similarity = user_loc_similarity.permute(1, 0)




        out, h = self.rnn(new_x_embedding, h)  # (seq_len, user_len, hidden_size)

        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
            for j in range(i + 1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t, user_len)  # (user_len, )
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)  # (user_len, 1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
                w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)  # (user_len, 1)
                sum_w += w_j
                out_w[i] += w_j * out[j]  # (user_len, hidden_size)
            out_w[i] /= sum_w
            
        out_pu = torch.zeros(seq_len, user_len, 2 *
                             self.hidden_size, device=x.device)
        
        # user_region_preference = self.cross_region_model(active_user, p_u)
        for i in range(seq_len):
            # (user_len, hidden_size * 2)
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)

        y_linear = self.fc(out_pu)  # (seq_len, user_len, loc_count)
        region_pred = self.fc_region(out_pu)


        # criterion = SimLoss()
        # loss1 = criterion(s_out1, t_out1, self.fsq_mob_adj.to(x.device))
        # loss2 = criterion(s_out2, t_out2, self.taxi_mob_adj.to(x.device))


        # return y_linear, h, loss1 + loss2
        return y_linear, h, region_pred
        # return y_linear, h





'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        # (1, 200, 10)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c




class MGFN(nn.Module):
    def __init__(self, graph_num, node_num, output_dim):
        super(MGFN, self).__init__()
        self.encoder = MobilityPatternJointLearning(graph_num=graph_num, node_num=node_num, output_dim=output_dim)
        self.decoder_s = nn.Linear(output_dim, output_dim)
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.feature = None
        self.name = "MGFN"

    def forward(self, x):
        # x = x.unsqueeze(0)
        self.feature = self.encoder(x)
        out_s = self.decoder_s(self.feature)
        out_t = self.decoder_t(self.feature)
        return out_s, out_t

    def out_feature(self, ):
        return self.feature



class MobilityPatternJointLearning(nn.Module):
    """
    input: (7, 180, 180)
    output: (180, 144)
    """
    def __init__(self, graph_num, node_num, output_dim):
        super(MobilityPatternJointLearning, self).__init__()
        self.graph_num = graph_num
        self.node_num = node_num
        self.num_multi_pattern_encoder = 3
        self.num_cross_graph_encoder = 1
        self.multi_pattern_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=node_num, nhead=4) for _ in range(self.num_multi_pattern_encoder)])
        self.cross_graph_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=node_num, nhead=4) for _ in range(self.num_cross_graph_encoder)])
        self.fc = DeepFc(self.graph_num*self.node_num, output_dim)
        # self.linear_out = nn.Linear(node_num, output_dim)
        self.para1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para1.data.fill_(0.7)
        self.para2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para2.data.fill_(0.3)
        # assert node_num % 2 == 0
        # self.s_linear = nn.Linear(node_num, int(node_num / 2))
        # self.o_linear = nn.Linear(node_num, int(node_num / 2))
        # self.concat = ConcatLinear(int(node_num / 2), int(node_num / 2), node_num)

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        out = x
        for multi_pattern in self.multi_pattern_blocks:
            out = multi_pattern(out)
        multi_pattern_emb = out
        out = out.transpose(0, 1)
        for cross_graph in self.cross_graph_blocks:
            out = cross_graph(out)
        out = out.transpose(0, 1)
        out = out*self.para2 + multi_pattern_emb*self.para1
        out = out.contiguous()
        out = out.view(-1, self.node_num*self.graph_num)
        out = self.fc(out)
        
        out = self.layer_norm(out)
        return out



class RegionFuse(nn.Module):
    def __init__(self, self_input_dim=10, cross_input_dim=10, output_dim=10) -> None:
        super().__init__()
        self.linear = nn.Linear(self_input_dim + cross_input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)





# class CrossRegionModel(nn.Module):
#     # 加载进预训练的self region和cross region embeddings，输入一个user，输出user对region的preference embedding
#     def __init__(self, user_embed_dim, self_region_embedding, cross_region_embedding, interact_graph, device='cuda:0') -> None:
#         super().__init__()
#         self.self_region_embedding = self_region_embedding
#         self.cross_region_embedding = cross_region_embedding
#         self.device = device
#         # User-region interact matrix
#         self.interact_graph = interact_graph
#         # Embedding dim
#         self.user_embed_dim = user_embed_dim
#         self.self_region_dim = self_region_embedding.weight.shape[-1]
#         self.cross_region_dim = cross_region_embedding.shape[-1]
#         # Network
#         # self.self_transfer_layer = nn.Linear(self.self_region_dim, self.user_embed_dim)
#         self.cross_transfer_layer = nn.Linear(self.cross_region_dim, self.user_embed_dim)
#         self.item_attention_layer = ItemAttentionLayer(self.user_embed_dim)
#         self.domain_attention_layer = DomainAttentionLayer(self.user_embed_dim)

#         self.norm = nn.LayerNorm(user_embed_dim)

#     def forward(self, user_idx, user_embedding):
#         # self_region_embedding = self.norm(self.self_transfer_layer(self.self_region_embedding.to(user_embedding.device)))
#         self_region_embedding = self.norm(self.self_region_embedding.weight.to(self.device))
#         cross_region_embedding = self.norm(self.cross_transfer_layer(self.cross_region_embedding.to(self.device)))
#         s_u = self.item_attention_layer(user_embedding, cross_region_embedding, self.interact_graph[user_idx].to(self.device))
#         z_u = self.domain_attention_layer(s_u, user_embedding, self_region_embedding)
#         return z_u
    
#     def output_self_region_embedding(self):
#         return self.norm(self.self_region_embedding.weight.to(self.device))

#     def output_cross_region_embedding(self):
#         return self.norm(self.cross_transfer_layer(self.cross_region_embedding.to(self.device)))



class CrossRegionModel(nn.Module):
    # 加载进预训练的self region和cross region embeddings，输入一个user，输出user对region的preference embedding
    def __init__(self, user_embed_dim, self_region_embedding, cross_region_embedding, interact_graph, device) -> None:
        super().__init__()
        self.self_region_embedding = self_region_embedding
        self.cross_region_embedding = cross_region_embedding
        self.device = device
        # User-region interact matrix
        self.interact_graph = interact_graph.to(device)
        # Embedding dim
        self.user_embed_dim = user_embed_dim
        self.self_region_dim = self_region_embedding.weight.shape[-1]
        self.cross_region_dim = cross_region_embedding.shape[-1]
        # Network
        # self.self_transfer_layer = nn.Linear(self.self_region_dim, self.user_embed_dim)
        self.cross_transfer_layer = nn.Linear(self.cross_region_dim, self.user_embed_dim)
        self.item_attention_layer = ItemAttentionLayer(self.user_embed_dim)
        self.domain_attention_layer = DomainAttentionLayer(self.user_embed_dim)

        # LayerNorm
        self.norm = nn.LayerNorm(user_embed_dim)

        self.fc = nn.Linear(2 * user_embed_dim, user_embed_dim)


    def forward(self, user_idx, region_idx):
        cross_region_embedding = self.norm(self.cross_transfer_layer(self.cross_region_embedding))
        # user-region preference
        self.self_region_preference = torch.einsum('ij,jk->ik', self.interact_graph, self.self_region_embedding.weight).to(self.device)
        # self.cross_region_preference = torch.einsum('ij,jk->ik', self.interact_graph, cross_region_embedding).to(self.device)

        # self_region_preference = self.self_region_preference[user_idx, :]
        # cross_region_preference = self.cross_region_preference[user_idx, :]

        # region_embedding = self.self_region_embedding.weight[region_idx, :]

        self_region_preference = self.self_region_embedding.weight[region_idx, :]

        # self.region_preference = self.item_attention_layer(region_embedding, region_preference)
        return self_region_preference
    
    # def output_self_region_embedding(self):
    #     return self.norm(self.self_region_embedding.weight.to(self.device))

    # def output_cross_region_embedding(self):
    #     return self.norm(self.cross_transfer_layer(self.cross_region_embedding.to(self.device)))

    def output_region_preference(self):
        return self.region_preference






class ItemAttentionLayer(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, key, query):
        '''
        key: (embed_dim, )
        query: (region_num, embed_dim)
        mask: (region_num, )
        '''
        a = self.linear(F.relu(torch.einsum('jk,ijk->ijk', key, query)).transpose(0, 1)).squeeze()
        alpha = self.softmax(a)
        ele_product = torch.einsum('ij,kij->kij', key, query)
        p = torch.einsum('ijk,ji->jk', ele_product, alpha)
        return p


class DomainAttentionLayer(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, s_u, p_u, query):
        '''
        s_u: (embed_dim, )
        p_u: (embed_dim, )
        query: (region_num, embed_dim)
        mask: (region_num, )
        '''
        s_u = s_u.view(-1)
        p_u = p_u.view(-1)

        query = self.norm(query)

        b_s = self.linear(F.relu(s_u * query)).view(-1)
        b_p = self.linear(F.relu(p_u * query)).view(-1)

        beta_s = torch.exp(b_s) / (torch.exp(b_s) + torch.exp(b_p))
        beta_p = torch.exp(b_p) / (torch.exp(b_s) + torch.exp(b_p))

        z_u = torch.einsum('i,j->ij', beta_s, s_u) + torch.einsum('i,j->ij', beta_p, p_u)

        return z_u

