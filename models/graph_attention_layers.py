# -*- coding=utf-8 -*-
# @Time:2025/4/3 10:50
# @Author:liutao
# @File:graph_attention_layers.py
# @Software:PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, args, in_features, out_features, edge_dim, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = args.dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = args.alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(args.device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features + edge_dim, 1))).to(args.device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.device=args.device
    def forward(self, h, adj, edge_attr=None):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh, edge_attr, adj)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        N = Wh.size(0)
        adj_matrix = torch.zeros((N, N), dtype=torch.long, device=h.device)
        unit_matrix = torch.eye(N, device=h.device, dtype=torch.long)
        adj_matrix += unit_matrix
        for i in range(adj.size(1)):
            source, target = adj[:, i]
            adj_matrix[source.item(), target.item()] = 1
            adj_matrix[target.item(), source.item()] = 1

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, edge_attr, adj):
        with torch.no_grad():
            N = Wh.size()[0]  # number of nodes
            Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
            Wh_repeated_alternating = Wh.repeat(N, 1)
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
            edge_attr_repeated = torch.zeros((N * N, self.in_features)).to(self.device)
            for i in range(adj.size(1)):
                source, target = adj[:, i]
                idx = source.item() * N + target.item()
                edge_attr_repeated[idx, :] = edge_attr[i, :]
            edge_attr_repeated.to(self.device)
            all_combinations_matrix = torch.cat([all_combinations_matrix, edge_attr_repeated], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features + edge_attr.shape[
            1] if edge_attr is not None else 2 * self.out_features)  # (N, N, 2 * out_features)


class GATLayer(nn.Module):
    def __init__(self, args, in_dim, out_dim, edge_dim):
        super(GATLayer, self).__init__()
        self.args = args
        self.gat_layers = []
        self.dropout = nn.Dropout(args.dropout)
        for k in range(args.gat_layers - 1):
            self.gat_layers.append(
                [GraphAttentionLayer(args, in_dim, out_dim, edge_dim, concat=True) for _ in range(args.num_heads)])
            in_dim = out_dim
        self.gat_out = [GraphAttentionLayer(args, in_dim, out_dim, edge_dim, concat=False) for _ in
                        range(args.num_heads)]

    def forward(self, input_feature, adj, edge_attr=None):
        for gat in self.gat_layers:
            out_feature = [g(input_feature, adj, edge_attr).unsqueeze(1) for g in gat]
            out_feature = torch.cat(out_feature, dim=1)
            out_feature = out_feature.mean(dim=1)
            out_feature = self.dropout(out_feature)
            input_feature = out_feature
        out_feature = [g(input_feature, adj, edge_attr).unsqueeze(1) for g in self.gat_out]
        out_feature = torch.cat(out_feature, dim=1)
        out_feature = out_feature.mean(dim=1)
        out_feature = F.relu(out_feature)
        return out_feature
