import os
import torch

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_67"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Input Metadata
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Paths (Explicit Versioning for Cache Safety)
CACHE_VERSION = "ahs_dfn_v1"
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, f"train_data_{CACHE_VERSION}.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, f"val_data_{CACHE_VERSION}.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, f"test_data_{CACHE_VERSION}.npz")

# Output Paths
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA PARAMETERS
# =============================================================================
SEQ_LEN = 107
SCORED_LEN = 68
NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Backbone
HIDDEN_DIM = 64
GROWTH_RATE = 64
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]
DROPOUT = 0.1

# Feedback Module
FEEDBACK_GROWTH_RATE = 16
FEEDBACK_LATENT_DIM = 32

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16  # Strictly set to 16 as per requirements
LR = 1e-3  # AdamW Learning Rate
EPOCHS = 50  # Maximum epochs
NUM_WORKERS = 2  # DataLoader workers
SEED = 42  # Reproducibility seed

# Scheduler & Early Stopping
PATIENCE = 10  # Early stopping patience
LR_FACTOR = 0.5  # ReduceLROnPlateau factor
LR_PATIENCE = 5  # ReduceLROnPlateau patience
MIN_LR = 1e-6  # Minimum learning rate

# =============================================================================
# SYSTEM & DEBUGGING
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug Flags
DEBUG = False  # Set to True to run on a small subset
DEBUG_SUBSET_SIZE = 100  # Number of samples for debugging
