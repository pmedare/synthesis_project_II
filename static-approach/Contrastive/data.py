# data.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import os

class EllipticDataset:
    def __init__(self, raw_dir, processed_dir, self_loop=True, reverse_edge=True):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.self_loop = self_loop
        self.reverse_edge = reverse_edge
        self.num_classes = 2  # licit and illicit
        
    def process(self):
        classesDF = pd.read_csv(os.path.join(self.raw_dir, "elliptic_txs_classes.csv"))
        edgesDF = pd.read_csv(os.path.join(self.raw_dir, "elliptic_txs_edgelist.csv"))
        featuresDF = pd.read_csv(os.path.join(self.raw_dir, "elliptic_txs_features.csv"), header=None)
        featuresDF.columns = ['txId', 'timestep'] + ['f' + str(i) for i in range(165)]
        classesDF['class'] = classesDF['class'].map({'2': 0, '1': 1, 'unknown': -1})
        featuresDF = featuresDF.merge(classesDF, on='txId')

        # Move features 'class' to first column
        cols = list(featuresDF.columns)
        cols = cols[:1] + [cols[-1]] + cols[1:-1]
        featuresDF = featuresDF[cols]

        # Filter for the last timestep
        featuresDF = featuresDF[featuresDF['timestep'] == 48]
        
        # Create graph
        g = dgl.graph((edgesDF['txId1'].values, edgesDF['txId2'].values))
        
        if self.self_loop:
            g = dgl.add_self_loop(g)
        
        if self.reverse_edge:
            g = dgl.add_reverse_edges(g)
        
        g.ndata['feat'] = torch.tensor(featuresDF.iloc[:, 2:-1].values, dtype=torch.float32)
        g.ndata['label'] = torch.tensor(featuresDF['class'].values, dtype=torch.int64)
        
        return g

def get_dataloader(raw_dir, processed_dir, batch_size=32, shuffle=True):
    dataset = EllipticDataset(raw_dir, processed_dir)
    graph = dataset.process()
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    mask = node_labels >= 0  # Only use labeled nodes
    
    features = node_features[mask]
    labels = node_labels[mask]
    
    data = [(features[i], labels[i]) for i in range(len(labels))]
    
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
