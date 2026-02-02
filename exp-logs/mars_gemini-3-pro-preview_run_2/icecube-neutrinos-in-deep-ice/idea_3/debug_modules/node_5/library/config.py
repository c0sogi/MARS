import os
import torch

# =============================================================================
# File Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for Idea 3 (TCB) specific artifacts (cache, checkpoints)
WORKING_DIR = "./working/idea_3"
# Final submission directory
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")

# Data Paths
SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# Model & Data Hyperparameters
# =============================================================================
SEED = 42

# Input Dimensions
# Fixed sequence length for the 1D CNN
SEQ_LEN = 128
# Channels: [x, y, z, time, charge, auxiliary]
INPUT_CHANNELS = 6

# =============================================================================
# Training Hyperparameters
# =============================================================================
# Batch size optimized for A100 GPU
BATCH_SIZE = 512
# Initial learning rate
LEARNING_RATE = 1e-3
# Maximum number of training epochs
EPOCHS = 20
# Early stopping patience
PATIENCE = 3
# Number of dataloader workers
NUM_WORKERS = 12

# Debugging / Development Controls
# Set to an integer (e.g., 50000) to limit dataset size for faster debugging
# Set to None to use the full dataset
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Normalization Statistics
# =============================================================================
# Statistics derived from the training set analysis for Standard Scaling (Z-score).
# Used for features: time, x, y, z.
# Note: 'charge' uses log-transformation, 'auxiliary' is boolean.
STATS = {
    "time_mean": 12972.3621,
    "time_std": 4430.3896,
    "x_mean": 8.8420,
    "x_std": 276.9620,
    "y_mean": -2.5228,
    "y_std": 263.5519,
    "z_mean": -92.6730,
    "z_std": 305.2266,
}


# =============================================================================
# Utility Functions
# =============================================================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    import random
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
