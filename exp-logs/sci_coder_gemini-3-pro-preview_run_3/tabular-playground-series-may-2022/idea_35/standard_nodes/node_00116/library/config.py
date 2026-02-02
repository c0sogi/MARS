import os
import torch

# ==========================================
# Global Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 2e-5

# ==========================================
# Compute Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Path Definitions
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_35"

# Ensure the working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)

# ==========================================
# Model Architecture Constants
# ==========================================
EMBEDDING_DIM = 16

# Deep Selective-Residual Parallel Ensemble (DSR-PE) Configuration
# 5 independent streams with heterogeneous capacity and regularization
STREAM_CONFIGS = [
    # Stream 1 (Anchor): Standard Funnel, Moderate Dropout
    {"hidden_layers": [512, 256, 128], "dropout": 0.20},
    # Stream 2 (Anchor): Standard Funnel, Moderate Dropout
    {"hidden_layers": [512, 256, 128], "dropout": 0.20},
    # Stream 3 (High Capacity): Wide Funnel, Higher Dropout
    {"hidden_layers": [1024, 512, 256], "dropout": 0.25},
    # Stream 4 (High Capacity): Wide Funnel, Higher Dropout
    {"hidden_layers": [1024, 512, 256], "dropout": 0.25},
    # Stream 5 (Conservative): Standard Funnel, High Dropout
    {"hidden_layers": [512, 256, 128], "dropout": 0.30},
]
