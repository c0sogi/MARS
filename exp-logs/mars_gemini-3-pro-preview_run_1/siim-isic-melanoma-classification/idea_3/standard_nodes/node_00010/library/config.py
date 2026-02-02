import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Caching Paths (for deterministic processing if needed)
CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

# =============================================================================
# COMPUTE & REPRODUCIBILITY
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Utilizing available vCPUs

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Backbone: Swin Transformer V2 Tiny
# 'swinv2_tiny_window16_256' is designed for 256x256 input
BACKBONE = "swinv2_tiny_window16_256"
IMAGE_SIZE = 256
NUM_CLASSES = 1  # Binary classification

# Tabular Fusion Settings
USE_META = True
TABULAR_COLS = ["age_approx", "sex", "anatom_site_general_challenge"]
# Dimensions for the MLP processing the tabular data before GLU fusion
TABULAR_HIDDEN_DIM = 128
TABULAR_OUT_DIM = 64

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
BATCH_SIZE = 64  # A100 (40GB) can handle Swin-Tiny with this batch size
EPOCHS = 15  # Sufficient for convergence with OneCycleLR
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
MAX_GRAD_NORM = 1.0

# OneCycleLR Scheduler Params
PCT_START = 0.1  # Warmup for 10% of training
DIV_FACTOR = 25.0
FINAL_DIV_FACTOR = 1000.0

# =============================================================================
# REGULARIZATION & AUGMENTATION
# =============================================================================
DROP_PATH_RATE = 0.1  # Stochastic depth rate
LABEL_SMOOTHING = (
    0.0  # Binary Cross Entropy usually works best without smoothing for AUC
)
USE_RANDOM_ERASING = True
RANDOM_ERASE_PROB = 0.25
USE_WEIGHTED_SAMPLER = True  # Handle class imbalance

# =============================================================================
# DEBUGGING
# =============================================================================
# Set DEBUG to True to run on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 1000
