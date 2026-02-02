import os
import torch
import random
import numpy as np

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Feature Engineering Hyperparameters
# ==========================================
# Directional Binning
AZIMUTH_SECTORS = 4  # North-East, South-East, South-West, North-West

# Signal Quality Stratification
QUALITY_STRATA = (
    2  # Stratum 1: High Quality (L5/E5a/B2a or Phase Lock), Stratum 2: Others
)

# Aggregation Statistics per Bin
# We compute these stats for both Cn0DbHz and SvElevationDegrees
STATS_PER_FEATURE = 4  # Mean, Std, Min, Max
BASE_FEATURES_PER_BIN = 2  # Cn0DbHz, SvElevationDegrees

# Global Context Features
# SatCount, RawPseudorangeUncertaintyMeters, SinAzCentroid, CosAzCentroid
GLOBAL_FEATURES = 4

# Calculate Total Input Channels for the 1D CNN
# Formula: (Sectors * Strata * Base_Features * Stats) + Global
# (4 * 2 * 2 * 4) + 4 = 68 Channels
INPUT_CHANNELS = (
    AZIMUTH_SECTORS * QUALITY_STRATA * BASE_FEATURES_PER_BIN * STATS_PER_FEATURE
) + GLOBAL_FEATURES

# Output: Delta East (Meters), Delta North (Meters)
OUTPUT_CHANNELS = 2

# Temporal Resolution
SAMPLING_RATE = 1  # Hz (1 sample per second)

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# 1D SE-ResUNet Settings
HIDDEN_DIM = 64  # Number of filters in the first encoder block
ENCODER_DEPTH = 4  # Depth of the U-Net
KERNEL_SIZE = 3  # Convolution kernel size
DROPOUT_RATE = 0.1  # Dropout rate

# Decimated Deep Supervision
# Auxiliary heads attached at these downsampling factors
AUXILIARY_SCALES = [2, 4, 8]
AUX_LOSS_WEIGHT = 0.3  # Weight for auxiliary loss terms

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience
WEIGHT_DECAY = 1e-4  # AdamW weight decay
GRAD_CLIP = 1.0  # Gradient clipping norm

# ==========================================
# Hardware & Reproducibility
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
SEED = 42

# ==========================================
# Debugging / Development
# ==========================================
# Set DEBUG to True to run on a small subset of data for rapid iteration
DEBUG = False
DEBUG_DRIVE_COUNT = 2  # Number of drives to use in debug mode
DEBUG_EPOCHS = 2  # Number of epochs in debug mode


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Apply seeding immediately upon import
set_seed(SEED)
