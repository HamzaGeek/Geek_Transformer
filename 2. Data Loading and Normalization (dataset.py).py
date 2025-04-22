# data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PDWDataset(Dataset):
    """
    Dataset for Radar Pulse Descriptor Words (PDWs).
    Each item is a pulse train containing multiple pulses and their emitter labels.
    Applies per-train normalization to features.
    """
    def __init__(self, data_dir=None, pulse_trains=None):
        """
        Initialize the dataset.
        :param data_dir: Directory containing pulse train files (each file = one pulse train).
        :param pulse_trains: Optionally, a pre-loaded list of (features, labels) pairs.
        """
        self.sequences = []
        self.labels = []
        if pulse_trains is not None:
            # If data is directly provided as a list of sequences
            for feats, labs in pulse_trains:
                self.sequences.append(np.array(feats, dtype=np.float32))
                self.labels.append(np.array(labs, dtype=np.int64))
        elif data_dir is not None:
            # Load all pulse train files from the directory
            for fname in sorted(os.listdir(data_dir)):
                if not fname.lower().endswith(".csv"):
                    continue  # assuming .csv files for data
                filepath = os.path.join(data_dir, fname)
                data = np.loadtxt(filepath, delimiter=',')
                # Assume last column is label, first 5 columns are features
                feats = data[:, :5].astype(np.float32)
                labs  = data[:, 5].astype(np.int64)
                self.sequences.append(feats)
                self.labels.append(labs)
        else:
            raise ValueError("Must provide either data_dir or pulse_trains")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns a tuple (features, labels) for the pulse train at index idx.
        Features are normalized per pulse train.
        """
        feats = np.copy(self.sequences[idx])
        labs  = np.copy(self.labels[idx])
        # Normalize features within this pulse train
        # 0: ToA, 1: Frequency, 2: PW, 3: Amplitude, 4: AoA (assuming AoA is last column)
        # ToA normalization (min-max scaling to [0,1])
        toa = feats[:, 0]
        min_toa = toa.min()
        max_toa = toa.max()
        if max_toa > min_toa:
            feats[:, 0] = (toa - min_toa) / (max_toa - min_toa)
        else:
            feats[:, 0] = 0.0  # If only one pulse, set ToA = 0.

        # Statistical normalization for frequency, PW, amplitude (indices 1,2,3 if AoA is index 4)
        for col in [1, 2, 3]:
            mean_val = feats[:, col].mean()
            std_val = feats[:, col].std()
            if std_val > 1e-6:
                feats[:, col] = (feats[:, col] - mean_val) / std_val
            else:
                # If no variation (std ~ 0), set feature to 0
                feats[:, col] = 0.0

        # AoA normalization (divide by 360° to range [0,1])
        aoa_col = 4
        feats[:, aoa_col] = feats[:, aoa_col] / 360.0

        # Convert to torch tensors
        feats_tensor = torch.from_numpy(feats)
        labs_tensor  = torch.from_numpy(labs)
        return feats_tensor, labs_tensor

def pad_collate_fn(batch):
    """
    Collate function to pad a batch of pulse trains to the same length.
    Returns padded_features, padded_labels, and a mask for padded positions.
    """
    # Get lengths of each sequence in the batch
    lengths = [seq.shape[0] for seq, _ in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Prepare padded tensors
    feat_dim = batch[0][0].shape[1]  # number of features (should be 5)
    padded_feats = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    padded_labels = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.int64)  # -1 for padding

    for i, (feats, labs) in enumerate(batch):
        seq_len = feats.shape[0]
        padded_feats[i, :seq_len, :] = feats
        padded_labels[i, :seq_len] = labs
    # Create a mask for padding: True for padded (to ignore in attention)
    # shape: [batch_size, max_len]
    pad_mask = (padded_labels == -1)
    return padded_feats, padded_labels, pad_mask
