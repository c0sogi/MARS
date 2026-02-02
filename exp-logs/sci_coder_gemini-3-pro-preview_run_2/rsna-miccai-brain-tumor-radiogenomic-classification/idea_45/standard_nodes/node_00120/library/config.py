import os
import torch

# -----------------------------------------------------------------------------
# Global Configuration & Reproducibility
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# File System Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Paths (Pre-generated)
METADATA_DIR = "./metadata"
METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
METADATA_VAL = os.path.join(METADATA_DIR, "val.csv")
METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

# Working Directory (Specific to Idea 45)
WORKING_DIR = "./working/idea_45"
CACHE_DIR = WORKING_DIR  # Directory for caching processed numpy arrays
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Submission
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Processing Parameters
# -----------------------------------------------------------------------------
IMG_SIZE = 224
ROI_DEPTH_MIN = 0.15
ROI_DEPTH_MAX = 0.85

# Modality Configuration
# The order is critical for the Grouped Convolution strategy in the model stem.
# Group 1: FLAIR (Filters 0-7)
# Group 2: T2w   (Filters 8-15)
# Group 3: T1w   (Filters 16-23)
# Group 4: T1wCE (Filters 24-31)
MODALITY_ORDER = ["FLAIR", "T2w", "T1w", "T1wCE"]

# Biologically-Adaptive Strides
# Defines the neighbor distance: [Anchor-Stride, Anchor, Anchor+Stride]
# FLAIR/T2w (Stride 5): Capture broad anatomical context (Edema/Fluid)
# T1w/T1wCE (Stride 2): Capture fine textural heterogeneity and structural baseline
STRIDES = {"FLAIR": 5, "T2w": 5, "T1w": 2, "T1wCE": 2}

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
BACKBONE = "efficientnet_b0"
INPUT_CHANNELS = 12  # 4 modalities * 3 slices per modality
GROUPS = 4  # Enforces modality isolation in the first layer
DROPOUT_RATE = 0.5  # Regularization for the classification head

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay per idea specs
PATIENCE = 5  # Early stopping patience

# Debugging / Development Control
DEBUG = False  # Set to True to limit dataset size for quick pipeline verification
MAX_SAMPLES = None  # If DEBUG is True, limits train/val to this many samples (e.g., 50)

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Optimized for the available 12 vCPUs


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def setup_directories():
    """
    Creates the necessary working and submission directories if they do not exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


def get_device():
    """
    Returns the configured PyTorch device.
    """
    return torch.device(DEVICE)
