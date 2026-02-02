import os
import torch

# =============================================================================
# Global Configuration & Hyperparameters
# =============================================================================

# -----------------------------------------------------------------------------
# File Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Specific File Paths
SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Processing Parameters
# -----------------------------------------------------------------------------
# Number of pulses to sample/pad per event for the PointNet-like architecture
# We select the top N_PULSES by charge.
N_PULSES = 128

# Batch size for training and validation
BATCH_SIZE = 256

# Number of workers for DataLoader
NUM_WORKERS = 4

# Debugging: Limit dataset size if set to an integer (e.g., 10000). Set to None for full training.
DEBUG_SUBSET_SIZE = None

# -----------------------------------------------------------------------------
# Model Architecture Parameters
# -----------------------------------------------------------------------------
# Input features: [x, y, z, time, charge, auxiliary]
INPUT_DIM = 6

# Hidden dimension for the MLP and Global Feature Vector
HIDDEN_DIM = 128

# Output dimension: 3D vector (nx, ny, nz)
OUTPUT_DIM = 3

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
LEARNING_RATE = 1e-3
EPOCHS = 20
SEED = 42
PATIENCE = 3  # For Early Stopping

# -----------------------------------------------------------------------------
# Compute Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
