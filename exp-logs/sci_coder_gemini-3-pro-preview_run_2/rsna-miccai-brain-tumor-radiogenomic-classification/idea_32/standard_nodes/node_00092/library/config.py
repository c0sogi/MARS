import os
import torch

# -----------------------------------------------------------------------------
# Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata paths (generated previously)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for artifacts
WORKING_DIR = "./working"
# Specific cache directory for this idea iteration
CACHE_DIR = os.path.join(WORKING_DIR, "idea_32")
os.makedirs(CACHE_DIR, exist_ok=True)

# Model checkpoints and submission
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Preprocessing & ROI Selection
# -----------------------------------------------------------------------------
IMG_SIZE = 224
NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w
NUM_SLICES = 3  # 3 slices per modality (Anchor - Stride, Anchor, Anchor + Stride)
STRIDE = 5  # Geometric stride for context

# Input channels = 4 modalities * 3 slices = 12 channels
IN_CHANNELS = NUM_MODALITIES * NUM_SLICES

# ROI Selection Constraints
ROI_MIN_DEPTH = 0.15
ROI_MAX_DEPTH = 0.85

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5

# -----------------------------------------------------------------------------
# System & Hardware
# -----------------------------------------------------------------------------
SEED = 42
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def get_config_dict():
    """Returns a dictionary of the configuration for logging purposes."""
    return {k: v for k, v in globals().items() if k.isupper()}
