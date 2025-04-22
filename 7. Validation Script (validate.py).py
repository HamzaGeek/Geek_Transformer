# validate.py
import torch
from torch.utils.data import DataLoader
import config
from data.dataset import PDWDataset, pad_collate_fn
from models.transformer_model import PulseEmbeddingTransformer
from utils.metrics import evaluate_clustering
from utils.clustering import cluster_embeddings

# Load validation dataset (or test dataset)
val_dataset = PDWDataset(data_dir=config.VAL_DATA_DIR)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_fn)

# Load the trained model
model = PulseEmbeddingTransformer().to(config.DEVICE)
model.load_state_dict(torch.load("trained_model.pth", map_location=config.DEVICE))
model.eval()

all_metrics = []
with torch.no_grad():
    for batch_feats, batch_labels, pad_mask in val_loader:
        batch_feats = batch_feats.to(config.DEVICE)
        pad_mask = pad_mask.to(config.DEVICE)
        true_labels = batch_labels.numpy().flatten()

        # Get embeddings for this pulse train
        embeddings = model(batch_feats, key_padding_mask=pad_mask)[0]  # (seq_len, EMBED_DIM)
        embeddings = embeddings.cpu().numpy()
        # Cluster embeddings
        pred_labels = cluster_embeddings(embeddings, min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE)
        # Evaluate clustering
        mask = true_labels != -1  # ignore padded
        metrics = evaluate_clustering(true_labels[mask], pred_labels[mask])
        all_metrics.append(metrics)
        # Print metrics for this pulse train (optional)
        print(f"Pulse train {len(all_metrics)}: ARI={metrics['ARI']:.3f}, AMI={metrics['AMI']:.3f}, V-measure={metrics['V-measure']:.3f}")

# Compute mean metrics over all pulse trains
if all_metrics:
    mean_ARI = sum(m["ARI"] for m in all_metrics) / len(all_metrics)
    mean_AMI = sum(m["AMI"] for m in all_metrics) / len(all_metrics)
    mean_V   = sum(m["V-measure"] for m in all_metrics) / len(all_metrics)
    print(f"Overall Clustering Performance - ARI: {mean_ARI:.3f}, AMI: {mean_AMI:.3f}, V-measure: {mean_V:.3f}")
