# Synthesis Project II - Anomaly Detection 

This repository contains the code and resources for the work titled "Fraud Node Classification" by _Pol Medina, Adrián García, Paula Feliu, Roger Garcia, and Joan Samper_, presented on _June 30, 2024._

## Overview

In this project, we present a study and comparison of different machine learning and deep learning models for fraud detection in payment networks. The main approaches include five well-known machine learning algorithms applied to a static scenario and a novel dynamic approach combining a graph neural network, a recurrent neural network, and a decision tree for node classification.

The project is centered on the Bitcoin payment network, though these methods can be easily extrapolated to other bank payment networks.

## Models

### Static Models

1. **Random Forest**
   - Two approaches: unbalanced and balanced datasets.
   
2. **Autoencoder**
   - Reconstruction-based anomaly detection.
   
3. **Contrastive Learning**
   - Self-supervised learning for representation learning.
   
4. **Graph Convolutional Neural Network (GCN)**
   - Node classification using graph-based features.
   
5. **Self-supervised Learning**
   - Leveraging unlabeled data for improved classification.

### Dynamic Model

- **EvolveGCN with Random Forest**
  - Combining GCNs with recurrent neural networks to capture temporal dynamics and classify nodes using a Random Forest classifier.

## Dataset

We used the Elliptic Bitcoin Dataset, an open-source graph dataset representing Bitcoin transactions with 200k nodes and 234k edges. The dataset includes features for each node, labeled as either licit, illicit, or unknown.
