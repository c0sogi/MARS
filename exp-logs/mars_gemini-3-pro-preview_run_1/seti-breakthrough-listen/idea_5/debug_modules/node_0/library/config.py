import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata CSV Paths
TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Directories
# idea_5 corresponds to the Time-Distributed ResNet50-GN experiment
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Configuration
# ==========================================
IMG_HEIGHT = 273
IMG_WIDTH = 256
NUM_FRAMES = 6
NUM_CHANNELS = 1  # Single channel per frame (spectrogram intensity)

# Input shape expected by the dataset __getitem__: (6, 1, 273, 256)
INPUT_SHAPE = (NUM_FRAMES, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

# ==========================================
# Model Configuration
# ==========================================
MODEL_NAME = "TimeDistributed_ResNet50_GN_FPN"
BACKBONE = "resnet50"

# Critical for small batch sizes: Replace BatchNorm with GroupNorm
USE_GROUP_NORM = True
GN_GROUPS = 32  # Number of groups for Group Normalization

# Feature Pyramid / Head Settings
FPN_CHANNELS = 256
DROPOUT_RATE = 0.5

# ==========================================
# Training Configuration
# ==========================================
SEED = 42

# Batch size set to 12 to fit ResNet50 in memory while relying on GroupNorm for stability
BATCH_SIZE = 12
EPOCHS = 15

# Optimizer settings (AdamW)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

# Scheduler settings (OneCycleLR)
MAX_LR = 1e-3
PCT_START = 0.3
DIV_FACTOR = 25.0
FINAL_DIV_FACTOR = 1000.0

# Early Stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0001

# ==========================================
# Compute Configuration
# ==========================================
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Debugging / Development
# ==========================================
# Toggle DEBUG to True to run a quick training loop on a small subset
DEBUG = False
DEBUG_SAMPLE_SIZE = 200  # Number of samples to use when DEBUG is True


def get_config_dict():
    """
    Returns the configuration as a dictionary.
    """
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
