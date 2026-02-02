import os
import torch

# -----------------------------------------------------------------------------
# Directory & File Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Metadata Files (Generated previously)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
ROI_CACHE_PATH = os.path.join(WORKING_DIR, "roi_cache_v2.parquet")

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
STRIDE = 5  # Distance between stacked slices (Anchor-5, Anchor, Anchor+5)
NUM_SLICES_PER_MODALITY = 3
# Total channels = 4 modalities * 3 slices = 12
IN_CHANNELS = len(MODALITIES) * NUM_SLICES_PER_MODALITY
NUM_CLASSES = 1

# ROI Selection Parameters
EXCLUDE_BOUNDARY_RATIO = 0.15  # Exclude top/bottom 15% of slices

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay as specified
PATIENCE = 5  # Early stopping patience

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_directories():
    """
    Creates necessary working and submission directories if they don't exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
