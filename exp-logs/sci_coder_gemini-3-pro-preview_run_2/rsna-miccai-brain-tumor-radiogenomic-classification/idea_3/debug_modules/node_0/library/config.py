import os
import torch

# -----------------------------------------------------------------------------
# Global Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata paths (already generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching and checkpoints
WORKING_DIR = "./working/idea_3"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output submission path
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_WORKERS = 2  # Adjusted for available vCPUs (12) but kept safe for stability

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 256
NUM_SLICES_PER_VIEW = 3
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
NUM_MODALITIES = len(MODALITIES)

# Total input channels = Slices per view * Modalities
# For the Siamese network, this is the channel count for ONE branch.
IN_CHANNELS = NUM_SLICES_PER_VIEW * NUM_MODALITIES  # 3 * 4 = 12

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
BACKBONE = "efficientnet_b0"
PRETRAINED = True
NUM_CLASSES = 1  # Binary classification (MGMT_value)

# -----------------------------------------------------------------------------
# Compute Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Caching Configuration
# -----------------------------------------------------------------------------
# Filenames for cached processed datasets to speed up subsequent runs
CACHE_TRAIN_FILE = os.path.join(WORKING_DIR, "train_cache.parquet")
CACHE_VAL_FILE = os.path.join(WORKING_DIR, "val_cache.parquet")
CACHE_TEST_FILE = os.path.join(WORKING_DIR, "test_cache.parquet")
