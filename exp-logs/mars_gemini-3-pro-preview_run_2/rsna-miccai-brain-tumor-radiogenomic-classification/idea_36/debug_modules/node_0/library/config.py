import os
import torch

# -----------------------------------------------------------------------------
# Global Configuration & Hyperparameters
# -----------------------------------------------------------------------------

# Reproducibility
SEED = 42

# Data Dimensions & Model Architecture
IMG_SIZE = 256
NUM_SLICES_PER_MODALITY = 3  # Stride strategy: [Anchor-5, Anchor, Anchor+5]
NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w
NUM_CHANNELS = NUM_MODALITIES * NUM_SLICES_PER_MODALITY  # Total input channels: 12

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.5

# Circuit Breaker Configuration
# If the ratio of failed subjects (e.g., missing keys/files) exceeds this threshold,
# the pipeline will raise a fatal error to prevent silent failure on zero-filled tensors.
CIRCUIT_BREAKER_THRESHOLD = 0.01

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# We have 12 vCPUs available, leaving some overhead for system processes
NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------

# Base Input Directories
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Files (Pre-generated in ./metadata)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Caching and Model Artifacts
# Using 'idea_36' as the specific iteration folder based on the task context
WORKING_DIR = "./working/idea_36"
os.makedirs(WORKING_DIR, exist_ok=True)

# Cache Paths for Pre-Cached Tensor Strategy
# These files store the processed numpy tensors to eliminate repeated I/O
CACHE_DIR = WORKING_DIR
TRAIN_CACHE_DATA = os.path.join(CACHE_DIR, "train_data.npy")
TRAIN_CACHE_LABELS = os.path.join(CACHE_DIR, "train_labels.npy")
VAL_CACHE_DATA = os.path.join(CACHE_DIR, "val_data.npy")
VAL_CACHE_LABELS = os.path.join(CACHE_DIR, "val_labels.npy")
TEST_CACHE_DATA = os.path.join(CACHE_DIR, "test_data.npy")
TEST_CACHE_IDS = os.path.join(CACHE_DIR, "test_ids.npy")

# Model Checkpointing
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Submission Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
