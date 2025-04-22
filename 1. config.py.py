# config.py
import torch

# Data paths (modify these to point to actual data files or directories)
TRAIN_DATA_DIR = "data/train"     # directory containing training pulse-train files
VAL_DATA_DIR   = "data/val"       # directory containing validation pulse-train files
TEST_DATA_DIR  = "data/test"      # directory for test or inference pulse-train files

# Hyperparameters and model config
NUM_LAYERS = 8            # Transformer layers
NUM_HEADS = 8             # Multi-head attention heads
D_MODEL = 256             # Transformer model (residual) dimension&#8203;:contentReference[oaicite:13]{index=13}
DIM_FEEDFORWARD = 2048    # Feed-forward network dimension&#8203;:contentReference[oaicite:14]{index=14}
EMBEDDING_DIM = 8         # Dimension of output embedding&#8203;:contentReference[oaicite:15]{index=15}
DROPOUT = 0.05            # Dropout rate&#8203;:contentReference[oaicite:16]{index=16}
TRIPLET_MARGIN = 1.9      # Triplet loss margin&#8203;:contentReference[oaicite:17]{index=17}

# Training parameters
BATCH_SIZE = 8            # Batch size&#8203;:contentReference[oaicite:18]{index=18}
EPOCHS = 8                # Number of training epochs&#8203;:contentReference[oaicite:19]{index=19}
LEARNING_RATE = 1e-4      # Learning rate for Adam optimizer&#8203;:contentReference[oaicite:20]{index=20}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clustering parameters
HDBSCAN_MIN_CLUSTER_SIZE = 20   # HDBSCAN minimum cluster size&#8203;:contentReference[oaicite:21]{index=21}

# Random seed for reproducibility (optional)
SEED = 42
