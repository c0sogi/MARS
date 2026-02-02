import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files (Pre-generated)
TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working Directory Structure
WORKING_DIR = "./working/idea_69"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

# Submission
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL HYPERPARAMETERS
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# Debugging / Development Flags
DEBUG = False
DEBUG_SUBSET_SIZE = 100  # Size of dataset subset when DEBUG is True

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
IMAGE_SIZE = 75
INPUT_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average
NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)

# =============================================================================
# MODEL ARCHITECTURE: MCI-CNN
# =============================================================================
# Backbone: Plain CNN (4 Stages)
# Width Strategy: Early Expansion (64 -> 128 -> 128 -> 128)
BACKBONE_CHANNELS = [64, 128, 128, 128]

# Activation Function
LEAKY_RELU_SLOPE = 0.1

# Attention Mechanism (SE Module)
SE_REDUCTION_RATIO = 16

# Readout (Corrected Decoupled Isomorphic)
# Projection dimension for Stage 3 and Stage 4 before pooling
READOUT_PROJ_DIM = 64
# Total Feature Dimension: (ProjDim * 2 pooling types) * 2 stages = 64*2 + 64*2 = 256
FEATURE_DIM = 256

# Multiplicative Calibration Head
CALIBRATION_HIDDEN_DIM = 16
# Output splits into Scale (256) and Shift (256) -> Total 512
CALIBRATION_OUTPUT_DIM = FEATURE_DIM * 2

# Classification Head
DROPOUT_RATE = 0.5

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
NUM_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 Regularization
NUM_EPOCHS = 75
PATIENCE = 12  # Early Stopping Patience


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def setup_directories():
    """
    Creates the necessary directory structure for the project.
    Ensures working, cache, checkpoint, and submission directories exist.
    """
    dirs_to_create = [WORKING_DIR, CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
    # print(f"Directories initialized: {dirs_to_create}")
