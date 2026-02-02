import os
import torch

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# File Paths
# ==========================================
INPUT_ROOT = "./input"
TRAIN_IMAGES_DIR = os.path.join(INPUT_ROOT, "train_images")
TEST_IMAGES_DIR = os.path.join(INPUT_ROOT, "test_images")

# Metadata
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working Directory (Cache & Models)
WORKING_DIR = "./working/idea_2"
CACHE_DIR = WORKING_DIR
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Submission
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
# Image Dimensions
IMG_SIZE = (224, 224)

# 2.5D Input: Stacking slices z-1, z, z+1
IN_CHANNELS = 3

# Sequence Length: Number of slices uniformly sampled per study
SEQ_LEN = 64

# Debugging
DEBUG = False
DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True

# ==========================================
# Model Configuration (Sequential 2.5D MIL)
# ==========================================
BACKBONE = "resnet18"
HIDDEN_DIM = 512  # Output dimension of the CNN backbone
LSTM_HIDDEN_DIM = 256  # Hidden dimension of the Bi-LSTM
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
BIDIRECTIONAL = True

# ==========================================
# Training Configuration
# ==========================================
# Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS
BATCH_SIZE = 4  # Physical batch size (fits in GPU memory)
ACCUMULATION_STEPS = 4  # Gradient accumulation steps
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 3  # Early stopping patience

# Hardware
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Target Labels
# ==========================================
TARGET_COLS = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]


# ==========================================
# Setup Utilities
# ==========================================
def setup_directories():
    """Creates necessary directories for outputs and cache."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
setup_directories()
