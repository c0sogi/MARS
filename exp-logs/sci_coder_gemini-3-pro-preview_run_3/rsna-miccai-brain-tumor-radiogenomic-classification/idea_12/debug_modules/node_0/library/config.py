import os
import torch

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Directory & File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Paths (Pre-generated)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
SUBMISSION_FILE = "submission.csv"
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Cache Paths for Deterministic Data Processing
CACHE_TRAIN_X = os.path.join(WORKING_DIR, "cached_train_X.npy")
CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "cached_train_y.npy")
CACHE_VAL_X = os.path.join(WORKING_DIR, "cached_val_X.npy")
CACHE_VAL_Y = os.path.join(WORKING_DIR, "cached_val_y.npy")
CACHE_TEST_X = os.path.join(WORKING_DIR, "cached_test_X.npy")
CACHE_TEST_IDS = os.path.join(WORKING_DIR, "cached_test_ids.npy")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# The model requires a specific input volume depth and resolution
NUM_SLICES = 32
IMG_SIZE = 256
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
NUM_MODALITIES = len(MODALITIES)

# The input tensor will have interleaved channels:
# [Slice0_FLAIR, Slice0_T1w, Slice0_T1wCE, Slice0_T2w, Slice1_FLAIR, ...]
# Total Input Channels = 32 slices * 4 modalities = 128
IN_CHANNELS = NUM_SLICES * NUM_MODALITIES

# ==========================================
# Model & Training Hyperparameters
# ==========================================
# Batch size adjusted for A100 40GB memory with large input tensors (128x256x256)
BATCH_SIZE = 16
NUM_EPOCHS = 10

# Optimization
LEARNING_RATE = 1e-4
# Explicitly set Weight Decay to 0.0 as per the "Idea" to avoid suppressing signal
WEIGHT_DECAY = 0.0

# Hardware Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
