# train.py
import torch
from torch.utils.data import DataLoader
import config
from data.dataset import PDWDataset, pad_collate_fn
from models.transformer_model import PulseEmbeddingTransformer
from losses.triplet_loss import batch_all_triplet_loss
from utils.metrics import evaluate_clustering
from utils.clustering import cluster_embeddings

# Set random seed for reproducibility (optional)
torch.manual_seed(config.SEED)

# Load training and validation datasets
train_dataset = PDWDataset(data_dir=config.TRAIN_DATA_DIR)
val_dataset   = PDWDataset(data_dir=config.VAL_DATA_DIR)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                          shuffle=True, collate_fn=pad_collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_fn)
# Note: For validation, we use batch_size=1 to evaluate one pulse train at a time (to cluster each independently).

# Initialize model, optimizer, and move model to GPU if available
model = PulseEmbeddingTransformer().to(config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

for epoch in range(1, config.EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch_feats, batch_labels, pad_mask in train_loader:
        batch_feats = batch_feats.to(config.DEVICE)
        batch_labels = batch_labels.to(config.DEVICE)
        pad_mask = pad_mask.to(config.DEVICE)

        # Forward pass to get embeddings
        embeddings = model(batch_feats, key_padding_mask=pad_mask)  # (B, L, EMBED_DIM)
        # Flatten the batch dimensions for loss computation
        B, L, EMBED_DIM = embeddings.shape
        embeddings_flat = embeddings.view(B * L, EMBED_DIM)
        labels_flat = batch_labels.view(B * L)
        # To avoid false positives across different pulse trains in the batch,
        # add an offset to labels of each sequence in the batch.
        # (This ensures that identical label numbers from different pulse trains are treated as distinct.)
        # We achieve this by adding a multiple of 10000 based on batch index within the flattening:
        # Compute sequence indices for each element in the flat batch
        seq_indices = torch.arange(B, device=config.DEVICE).repeat_interleave(L)
        labels_flat = labels_flat + seq_indices * 10000  # large offset to separate label namespaces

        # Compute triplet loss
        loss = batch_all_triplet_loss(embeddings_flat, labels_flat, margin=config.TRIPLET_MARGIN)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{config.EPOCHS} - Training Loss: {avg_loss:.4f}")

    # Validation: evaluate clustering performance on validation set
    model.eval()
    if len(val_dataset) > 0:
        all_ari, all_ami, all_v = [], [], []
        with torch.no_grad():
            for batch_feats, batch_labels, pad_mask in val_loader:
                # Each batch here is a single pulse train (batch_size=1)
                batch_feats = batch_feats.to(config.DEVICE)
                pad_mask = pad_mask.to(config.DEVICE)
                batch_labels = batch_labels.numpy().flatten()  # true labels for this pulse train (CPU numpy)

                # Get embeddings for the pulse train
                embeddings = model(batch_feats, key_padding_mask=pad_mask)[0]  # shape (seq_len, EMBED_DIM) for this train
                embeddings = embeddings.cpu().numpy()
                # Cluster the embeddings with HDBSCAN
                pred_labels = cluster_embeddings(embeddings, min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE)
                # Compute metrics (ignoring noise labels if any)
                metrics = evaluate_clustering(batch_labels[batch_labels != -1], pred_labels[batch_labels != -1])
                all_ari.append(metrics["ARI"]); all_ami.append(metrics["AMI"]); all_v.append(metrics["V-measure"])
        # Average metrics over all pulse trains in validation set
        mean_ari = sum(all_ari) / len(all_ari)
        mean_ami = sum(all_ami) / len(all_ami)
        mean_v   = sum(all_v)   / len(all_v)
        print(f"Validation - ARI: {mean_ari:.3f}, AMI: {mean_ami:.3f}, V-measure: {mean_v:.3f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved to trained_model.pth")
