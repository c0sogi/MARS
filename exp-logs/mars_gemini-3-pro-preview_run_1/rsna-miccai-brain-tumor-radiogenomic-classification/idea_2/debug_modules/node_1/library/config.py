import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
# Root directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output File Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Cache Paths (for pre-processed tensors if needed)
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.parquet")

# ==========================================
# Data Hyperparameters
# ==========================================
# MIL Bag Configuration
NUM_SLICES = 32  # Number of slices to sample per subject (Bag size)
IMAGE_SIZE = 256  # Input image spatial dimension (256x256)
CHANNELS = 3  # Number of channels (FLAIR, T1wCE, T2w)

# Preprocessing
REMOVE_NOISE_MARGIN = 0.1  # Fraction of top/bottom volume to discard before sampling

# ==========================================
# Model Hyperparameters
# ==========================================
BACKBONE = "efficientnet_b0"
PRETRAINED = True
HIDDEN_DIM = 256  # Dimension for the attention mechanism
DROPOUT_RATE = 0.5  # Dropout probability

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 8  # Number of subjects (bags) per batch
NUM_EPOCHS = 15  # Total training epochs
LEARNING_RATE = 1e-4  # AdamW learning rate
WEIGHT_DECAY = 1e-2  # AdamW weight decay
PATIENCE = 5  # Early stopping patience

# ==========================================
# System & Reproducibility
# ==========================================
SEED = 42
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Debugging
# ==========================================
# Toggle to run on a small subset of data
DEBUG = False
DEBUG_SAMPLE_SIZE = 16
