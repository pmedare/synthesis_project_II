# loss.py
import torch
import torch.nn.functional as F

def contrastive_loss(anchor, positive, negative, wp, wn, margin):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    loss = torch.mean(F.relu(wp * pos_dist - wn * (margin - neg_dist)))
    return loss
