import os
import torch

# ==========================================
# Directories and File Paths
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata paths (generated previously)
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Working directory for this specific idea
WORKING_DIR = "./working/idea_8"
os.makedirs(WORKING_DIR, exist_ok=True)

# Cache paths for deterministic data loading
CACHE_DIR = WORKING_DIR
TRAIN_CACHE_X = os.path.join(CACHE_DIR, "cached_train_X.npy")
TRAIN_CACHE_Y = os.path.join(CACHE_DIR, "cached_train_y.npy")
VAL_CACHE_X = os.path.join(CACHE_DIR, "cached_val_X.npy")
VAL_CACHE_Y = os.path.join(CACHE_DIR, "cached_val_y.npy")
TEST_CACHE_X = os.path.join(CACHE_DIR, "cached_test_X.npy")
TEST_CACHE_IDS = os.path.join(CACHE_DIR, "cached_test_ids.npy")

# Model and Submission paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Model Hyperparameters
# ==========================================
IMG_SIZE = 256  # Spatial resolution (H, W)
NUM_SLICES = 32  # Number of slices per modality to sample
NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w
# Total input channels = NUM_SLICES * NUM_MODALITIES
INPUT_CHANNELS = NUM_SLICES * NUM_MODALITIES  # 128

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
SEED = 42

# Early Stopping parameters
PATIENCE = 3
MIN_DELTA = 1e-4

# ==========================================
# System Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of subprocesses for data loading
