import torch
import torch.nn as nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv
from torch.nn.parameter import Parameter

class MatGRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Sigmoid())
        self.reset = MatGRUGate(in_feats,
                                out_feats,
                                torch.nn.Sigmoid())
        self.htilda = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

class MatGRUGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.Tensor(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)
        return out

class TopK(torch.nn.Module):
    """
    Similar to the official `egcn_h.py`. We only consider the node in a timestamp based subgraph,
    so we need to pay attention to `K` should be less than the min node numbers in all subgraph.
    Please refer to section 3.4 of the paper for the formula.
    """
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_parameters()
        self.k = k

    def reset_parameters(self):
        init.xavier_uniform_(self.scorer)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        vals, topk_indices = scores.view(-1).topk(self.k)
        out = node_embs[topk_indices] * torch.tanh(scores[topk_indices].view(-1, 1))
        # we need to transpose the output
        return out.t()

class EvolveGCNH(nn.Module):
    def __init__(self, in_feats=166, n_hidden=76, num_layers=2, n_classes=2, classifier_hidden=510):
        # default parameters follow the official config
        super(EvolveGCNH, self).__init__()
        self.num_layers = num_layers
        self.pooling_layers = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        self.pooling_layers.append(TopK(in_feats, n_hidden))
        # similar to EvolveGCNO
        self.recurrent_layers.append(MatGRUCell(in_feats=in_feats, out_feats=n_hidden))
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
        self.gnn_convs.append(
            GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))
        for _ in range(num_layers - 1):
            self.pooling_layers.append(TopK(n_hidden, n_hidden))
            self.recurrent_layers.append(MatGRUCell(in_feats=n_hidden, out_feats=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))

        self.reset_parameters()

    def reset_parameters(self):
        for gcn_weight in self.gcn_weights_list:
            init.xavier_uniform_(gcn_weight)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata['feat'])
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                X_tilde = self.pooling_layers[i](feature_list[j])
                W = self.recurrent_layers[i](W, X_tilde)
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W)
        return feature_list[-1]

class EvolveGCNO(nn.Module):
    def __init__(self, in_feats=166, n_hidden=256, num_layers=2, n_classes=2, classifier_hidden=307):
        # default parameters follow the official config
        super(EvolveGCNO, self).__init__()
        self.num_layers = num_layers
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        self.recurrent_layers.append(MatGRUCell(in_feats=in_feats, out_feats=n_hidden))
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
        self.gnn_convs.append(
            GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))
        for _ in range(num_layers - 1):
            self.recurrent_layers.append(MatGRUCell(in_feats=n_hidden, out_feats=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))

        self.reset_parameters()

    def reset_parameters(self):
        for gcn_weight in self.gcn_weights_list:
            init.xavier_uniform_(gcn_weight)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata['feat'])
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                W = self.recurrent_layers[i](W)
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W)
        return feature_list[-1]
