import os
import torch
import numpy as np
import random

# =============================================================================
# 1. File Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Specific File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")
SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# 2. Data Hyperparameters
# =============================================================================
SEQ_LEN = 128  # Fixed number of pulses per event
N_FEATURES = 6  # Features: [x, y, z, time, charge, auxiliary]

# Column definitions
ID_COL = "event_id"
TARGET_COLS = ["azimuth", "zenith"]

# =============================================================================
# 3. Normalization Statistics (Derived from Data Analysis)
# =============================================================================
# Used for Standard Scaling: (x - mean) / std
STATS = {
    "time_mean": 12972.36,
    "time_std": 4430.39,
    "x_mean": 8.84,
    "x_std": 276.96,
    "y_mean": -2.52,
    "y_std": 263.55,
    "z_mean": -92.67,
    "z_std": 305.23,
    # Charge is log-transformed, so we don't use standard scaling stats here directly
    # but we might normalize after log transform if needed.
    # For now, we assume log1p(charge) is sufficient or handled in dataset.
}

# =============================================================================
# 4. Model Hyperparameters
# =============================================================================
EMBED_DIM = 64  # Dimension of the initial feature projection
HIDDEN_DIM = 128  # Hidden dimension of the GRU
NUM_LAYERS = 2  # Number of GRU layers
DROPOUT = 0.1  # Dropout rate
OUTPUT_DIM = 3  # Output vector (x, y, z) direction

# =============================================================================
# 5. Training Hyperparameters
# =============================================================================
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
NUM_WORKERS = 4  # Number of dataloader workers
SEED = 42

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 6. Utility Functions
# =============================================================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
