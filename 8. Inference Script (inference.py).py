# inference.py
import torch
import numpy as np
import config
from models.transformer_model import PulseEmbeddingTransformer
from utils.clustering import cluster_embeddings
from data.dataset import PDWDataset, pad_collate_fn

# Load the trained model
model = PulseEmbeddingTransformer().to(config.DEVICE)
model.load_state_dict(torch.load("trained_model.pth", map_location=config.DEVICE))
model.eval()

# Load the test data (assuming a directory or a specific file for inference)
test_dataset = PDWDataset(data_dir=config.TEST_DATA_DIR)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_fn)

with torch.no_grad():
    for i, (feats, _, pad_mask) in enumerate(test_loader, start=1):
        # feats: (1, seq_len, 5), pad_mask: (1, seq_len)
        feats = feats.to(config.DEVICE)
        pad_mask = pad_mask.to(config.DEVICE)
        # Get embeddings for this pulse train
        embeddings = model(feats, key_padding_mask=pad_mask)[0]  # (seq_len, EMBED_DIM)
        embeddings = embeddings.cpu().numpy()
        # Cluster the embeddings
        cluster_labels = cluster_embeddings(embeddings, min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE)
        # Print or save results
        print(f"Pulse train {i}: {len(cluster_labels)} pulses, clustered into {len(set(cluster_labels[cluster_labels>=0]))} emitters")
        # For example, print first 10 pulses with their cluster assignments
        for j, lbl in enumerate(cluster_labels[:10]):
            print(f" Pulse {j}: cluster {lbl}")
        # (Here, -1 labels from HDBSCAN indicate noise/outliers that didn't fit well into any cluster)
