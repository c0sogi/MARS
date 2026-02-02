import os
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"

# Input Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")


def setup_directories():
    """
    Creates the necessary working directories for checkpoints, cache, and submissions.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================

SEED = 42
ORIGINAL_IMG_SIZE = 75
IMG_SIZE = 224  # Upsampled via Bicubic Interpolation
NUM_CHANNELS = 3  # Band 1 (Norm), Band 2 (Norm), Composite (Avg)
NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)

# Debugging
DEBUG = False
DEBUG_SAMPLES = 100

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

BACKBONE = "resnet18"
PRETRAINED = True
DROPOUT_RATE = 0.5
GEM_P_INIT = 1.0  # Initial GeM pooling parameter (starts as Average Pooling)
GEM_P_TRAINABLE = True

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 32

# Optimization (SAM + AdamW)
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05

# Scheduler (ReduceLROnPlateau)
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
SCHEDULER_MIN_LR = 1e-6

# Training Loop
N_FOLDS = 5
MAX_EPOCHS_PHASE_1 = 60  # Upper bound for calibration phase
SWA_EPOCHS = 12  # Phase 2 SWA duration
EARLY_STOPPING_PATIENCE = 15

# =============================================================================
# INFERENCE HYPERPARAMETERS
# =============================================================================

TTA_ENABLED = True
TTA_STEPS = 4  # Klein Four-Group: Original, HFlip, VFlip, Rot180
