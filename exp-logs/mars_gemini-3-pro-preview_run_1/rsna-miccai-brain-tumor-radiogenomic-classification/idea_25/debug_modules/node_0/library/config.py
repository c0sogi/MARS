import os
import torch

# ==========================================
# System & Reproducibility
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # As per compute availability

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_25")
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Hyperparameters
# ==========================================
# Modalities to use for the 9-channel stack (Order matters for channel mapping)
MODALITIES = ["FLAIR", "T1wCE", "T2w"]

IMG_SIZE = 224
STRIDE = 5  # Delta for volumetric slicing (z - delta, z, z + delta)
INPUT_CHANNELS = 9  # 3 modalities * 3 slices

# Excluded cases (already handled in metadata, but kept for reference)
EXCLUDE_CASES = [109, 123, 709]

# ==========================================
# Model & Training Hyperparameters
# ==========================================
BACKBONE = "efficientnet_b0"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.3
NUM_FOLDS = 5
EPOCHS = 15

# ==========================================
# Debugging & Development
# ==========================================
# Set to True to run on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_DATASET_SIZE = 50
