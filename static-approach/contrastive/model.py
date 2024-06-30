# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim, activation=F.relu)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = self.conv2(g, x)
        return x
