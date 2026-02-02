import os
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
IDEA_ID = "idea_25"
BASE_DIR = "."
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working", IDEA_ID)
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CHECKPOINT_DIR = WORKING_DIR

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMAGE_SIZE = (32, 32)
NUM_CLASSES = 1
NUM_WORKERS = 2  # Adjust based on vCPU availability (12 vCPUs available)
PIN_MEMORY = True

# Debugging / Development
DEBUG = False
DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use when DEBUG is True

# =============================================================================
# MODEL ARCHITECTURE (Custom Wide SE-Res2NeXt)
# =============================================================================
# Backbone: 3-stage (32x32 -> 16x16 -> 8x8)
# "Super-Wide" Channel Configuration: [64, 128, 256]
MODEL_PARAMS = {
    "stages": [64, 128, 256],
    "cardinality": 32,  # Groups for ResNeXt
    "base_width": 4,  # Width per group (can be inferred or explicit)
    "res2net_scale": 4,  # Scale factor for Res2Net hierarchical connections
    "se_reduction": 16,  # Reduction ratio for Squeeze-and-Excitation
    "dropout_rate": 0.0,  # Dropout for the final dense layer
    "use_gap": True,  # Use Global Average Pooling (vs Covariance)
}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEEDS = [0, 1, 2, 3, 4]  # Homogeneous Seed Averaging
EPOCHS = 20
BATCH_SIZE = 128

# Optimizer (AdamW)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2  # Slightly higher for AdamW

# Scheduler (Cosine Annealing)
T_MAX = EPOCHS
ETA_MIN = 1e-6

# Early Stopping
PATIENCE = 5
MIN_DELTA = 1e-4

# System
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# AUGMENTATION STRATEGY
# =============================================================================
# Strictly "light augmentation"
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "rotate": False,
    "color_jitter": False,
    "cutout": False,
}

# Test Time Augmentation (TTA)
USE_TTA = True  # If True, averages predictions of Original, H-Flip, V-Flip
