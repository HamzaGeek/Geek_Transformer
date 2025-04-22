# losses/triplet_loss.py
import torch
import torch.nn.functional as F

def batch_all_triplet_loss(embeddings, labels, margin):
    """
    Compute the batch-all triplet loss for a batch of embeddings.
    :param embeddings: Tensor of shape (N, embed_dim) with all embeddings in the batch.
    :param labels: Tensor of shape (N,) with integer labels for each embedding (-1 for padding entries).
    :param margin: Float, margin for triplet loss.
    :return: Triplet loss (scalar tensor).
    """
    # Filter out padded entries (labels == -1)
    valid_mask = labels != -1
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]
    if embeddings.shape[0] == 0:
        # No valid pulses in this batch
        return torch.tensor(0.0, device=embeddings.device)
    
    # Compute pairwise distance matrix (Euclidean distance)
    # shape: [N, N]
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    N = dist_matrix.size(0)
    # Create masks for positive and negative pairs
    labels_mat = labels.unsqueeze(0) == labels.unsqueeze(1)   # shape [N, N], True where labels match
    # Exclude self-pairs from positive mask
    pos_mask = labels_mat.clone()
    pos_mask.fill_diagonal_(False)
    # Negative mask: labels differ
    neg_mask = ~labels_mat

    # Compute triplet loss for all i,j,k:
    # Expand distance matrix for broadcasting
    dist_anchor_pos = dist_matrix.unsqueeze(2)  # shape [N, N, 1]
    dist_anchor_neg = dist_matrix.unsqueeze(1)  # shape [N, 1, N]
    # Triplet condition: want dist(anchor, pos) + margin < dist(anchor, neg)
    triplet_losses = dist_anchor_pos - dist_anchor_neg + margin  # shape [N, N, N]
    # Only consider valid triplets (i,j positive and i,k negative)
    mask_anchor_pos = pos_mask.unsqueeze(2)  # anchor i, positive j
    mask_anchor_neg = neg_mask.unsqueeze(1)  # anchor i, negative k
    valid_triplet_mask = mask_anchor_pos & mask_anchor_neg  # shape [N, N, N]
    # Apply ReLU to get max(0, ...). This zeroes out easy triplets automatically.
    triplet_losses = F.relu(triplet_losses)
    # Filter out only the triplets that are valid
    valid_losses = triplet_losses[valid_triplet_mask]
    if valid_losses.numel() == 0:
        # No valid (non-easy) triplets in this batch
        return torch.tensor(0.0, device=embeddings.device)
    # Take average over all non-zero (non-easy) triplet losses&#8203;:contentReference[oaicite:32]{index=32}
    loss = valid_losses.mean()
    return loss
