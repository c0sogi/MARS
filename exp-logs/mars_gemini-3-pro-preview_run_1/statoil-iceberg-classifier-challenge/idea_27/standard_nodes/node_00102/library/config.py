import os
import torch

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_27"

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Cache Files
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.npz")

# Checkpoint Directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMG_SIZE = 224  # Upsampled from 75x75
RAW_IMG_SIZE = 75
NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)
INPUT_CHANNELS = 3  # 3-channel composite (Band1, Band2, Derived)

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
BACKBONE = "resnet18"
DROPOUT_RATE = 0.5
USE_LATE_FUSION = True
FUSION_DIM = 512  # ResNet18 GAP output dimension

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2

# Optimizer settings
LR = 2e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05

# Phase 1: Calibration (Scheduler settings)
PATIENCE = 10
FACTOR = 0.5
CALIBRATION_EPOCHS = 50  # Upper bound for finding convergence

# Phase 2: Production (SWA settings)
SWA_LR = 2e-4
SWA_EPOCHS = 12  # Number of epochs to run SWA after main training
SWA_START_EPOCH = None  # To be determined dynamically based on Phase 1

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Variant A: [Band 1, Band 2, Mean]
# Variant B: [Band 1, Band 2, Difference]
ENSEMBLE_CONFIG = [
    {"variant": "A", "seed": 42},
    {"variant": "A", "seed": 43},
    {"variant": "A", "seed": 44},
    {"variant": "B", "seed": 45},
    {"variant": "B", "seed": 46},
]

# =============================================================================
# HARDWARE
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
