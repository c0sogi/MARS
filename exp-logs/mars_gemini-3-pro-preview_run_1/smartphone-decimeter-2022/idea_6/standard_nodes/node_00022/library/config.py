import os
import torch
import numpy as np
import random

# =============================================================================
# 1. PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"  # Cache directory for processed features
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# 2. DATA PROCESSING CONFIGURATION
# =============================================================================
SEED = 42
SAMPLING_RATE = 1  # Hz (Align to 1 second intervals)

# Raw GNSS columns to aggregate per timestamp
RAW_GNSS_COLS = ["Cn0DbHz", "SvElevationDegrees", "RawPseudorangeUncertaintyMeters"]

# Aggregation statistics to compute for each raw column
# This implements the "Distribution-Aware Aggregation" from the idea
AGGREGATION_MAP = {
    "Cn0DbHz": ["mean", "max", "min"],
    "SvElevationDegrees": ["mean", "max", "min"],
    "RawPseudorangeUncertaintyMeters": ["mean"],
}

# Derived features to include
DERIVED_COLS = ["sat_count"]

# Calculate Input Dimension
# 3 (Cn0) + 3 (Elev) + 1 (Uncertainty) + 1 (SatCount) = 8
INPUT_DIM = sum(len(stats) for stats in AGGREGATION_MAP.values()) + len(DERIVED_COLS)

# Target Definitions
# We predict residuals in meters (East, North) relative to WLS baseline
TARGET_COLS = ["dLat_meters", "dLon_meters"]
OUTPUT_DIM = 2

# Device Context Mapping
# Maps phone model strings to integer indices for embedding
PHONE_NAME_TO_IDX = {
    "Pixel4": 0,
    "GooglePixel4": 0,
    "Pixel4XL": 1,
    "GooglePixel4XL": 1,
    "Pixel5": 2,
    "GooglePixel5": 2,
    "SamsungGalaxyS20Ultra": 3,
    "XiaomiMi8": 4,
    "Mi8": 4,
}
NUM_DEVICES = 5

# =============================================================================
# 3. MODEL HYPERPARAMETERS
# =============================================================================
# Encoder (1D ResNet)
ENCODER_CHANNELS = [32, 64, 128, 256]
KERNEL_SIZE = 3

# Bottleneck (Transformer)
TRANSFORMER_DIM = 256  # Must match last encoder channel width
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_FF_DIM = 512
DROPOUT = 0.1

# Context Embedding
DEVICE_EMBEDDING_DIM = 16

# =============================================================================
# 4. TRAINING SETTINGS
# =============================================================================
BATCH_SIZE = 4  # Number of full drives/sequences per batch
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 4


# =============================================================================
# 5. UTILITIES
# =============================================================================
def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize seed immediately upon import
set_seed(SEED)
