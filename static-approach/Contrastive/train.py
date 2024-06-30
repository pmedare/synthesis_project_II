import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import dgl
from data import get_dataloader
from model import GCN
from loss import contrastive_loss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def train(args, device):
    # Prepare data
    dataloader = get_dataloader(args.raw_dir, args.processed_dir, batch_size=args.batch_size)

    # Initialize model
    input_dim = 165  # Number of features
    hidden_dim = 128
    output_dim = 16  # Embedding dimension
    model = GCN(input_dim, hidden_dim, output_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize KNN for evaluation
    knn = KNeighborsClassifier(n_neighbors=5)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            g = dgl.graph((torch.arange(features.size(0)), torch.arange(features.size(0)))).to(device)
            embeddings = model(g, features)

            # Generate triplets
            anchor, positive, negative = [], [], []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] == labels[j]:
                        anchor.append(embeddings[i])
                        positive.append(embeddings[j])
                    else:
                        negative.append(embeddings[j])
            if not anchor or not positive or not negative:
                continue

            anchor = torch.stack(anchor)
            positive = torch.stack(positive)
            negative = torch.stack(negative)

            loss = contrastive_loss(anchor, positive, negative, args.wp, args.wn, args.margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{args.num_epochs}, Loss: {total_loss/len(dataloader)}')

        # Evaluate using KNN
        model.eval()
        all_features, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                g = dgl.graph((torch.arange(features.size(0)), torch.arange(features.size(0)))).to(device)
                embeddings = model(g, features)
                all_features.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_features = np.concatenate(all_features)
        all_labels = np.concatenate(all_labels)
        knn.fit(all_features, all_labels)
        predictions = knn.predict(all_features)
        report = classification_report(all_labels, predictions)
        print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN with Contrastive Learning on Elliptic Data')
    parser.add_argument('--raw_dir', type=str, required=True, help='Raw data directory')
    parser.add_argument('--processed_dir', type=str, required=True, help='Processed data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wp', type=float, default=1.0, help='Weight for positive samples')
    parser.add_argument('--wn', type=float, default=1.0, help='Weight for negative samples')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for contrastive loss')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args, device)
