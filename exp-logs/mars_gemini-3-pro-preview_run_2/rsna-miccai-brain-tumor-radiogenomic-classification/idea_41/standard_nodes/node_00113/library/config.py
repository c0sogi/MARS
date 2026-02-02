import os
import torch

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Random Seed for Reproducibility
SEED = 42

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of DataLoader Workers
# With 12 vCPUs, 4 workers is a safe balance to avoid overhead while maintaining throughput
NUM_WORKERS = 4

# =============================================================================
# FILE PATHS
# =============================================================================

# Input Directories (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working Directory (for caching and artifacts)
# This is where processed numpy/parquet files and model checkpoints will be stored
WORKING_DIR = "./working/idea_41"
os.makedirs(WORKING_DIR, exist_ok=True)

# Cache Directory for processed data
CACHE_DIR = WORKING_DIR

# Model Checkpoint Path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Submission Output
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA PROCESSING HYPERPARAMETERS
# =============================================================================

# Image Dimensions
# We resize to 224x224 using Area Interpolation
IMG_SIZE = 224
INPUT_SIZE = (IMG_SIZE, IMG_SIZE)

# Modalities
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

# Region of Interest (ROI) Selection
# We use FLAIR as the geometric anchor based on Raw Pixel Integral
ROI_ANCHOR_MODALITY = "FLAIR"
ROI_MIN = 0.15  # Minimum depth percentage (15%)
ROI_MAX = 0.85  # Maximum depth percentage (85%)

# Volumetric Slab Averaging Configuration
# We extract 3 slabs centered at [Anchor-5, Anchor, Anchor+5]
# Each slab is an average of 3 slices (Center-1, Center, Center+1)
NUM_SLABS = 3
SLAB_OFFSETS = [-5, 0, 5]  # Relative slice indices from the ROI anchor
SLAB_THICKNESS = 1  # Cite solution_lesson_node_00112: Concatenation > Averaging

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

BACKBONE = "efficientnet_b0"

# Input Channels = 4 Modalities * 3 Slabs = 12 Channels
INPUT_CHANNELS = len(MODALITIES) * NUM_SLABS

# Grouped Convolutions
# We use 4 groups in the stem to isolate the 4 modalities
GROUPS = len(MODALITIES)

# Regularization
DROPOUT_RATE = 0.5

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay
MAX_EPOCHS = 20
PATIENCE = 5  # Early stopping patience

# Augmentation
# Random Rotation limited to +/- 15 degrees
ROTATION_DEGREES = 15

# =============================================================================
# DEBUGGING
# =============================================================================

# Set to an integer (e.g., 50) to limit dataset size for quick debugging loops
# Set to None for full training
DEBUG_DATA_LIMIT = None
