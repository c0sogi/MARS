import os
import torch

# ==========================================
# Directories & Paths
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working directory for caching and model checkpoints
WORKING_DIR = "./working/idea_7"
os.makedirs(WORKING_DIR, exist_ok=True)

# ==========================================
# Data Processing Configuration
# ==========================================
# Image dimensions
SLICE_SIZE = 224  # Size of individual slice (H, W)
GRID_SIZE = 1  # Single slice to avoid montage artifacts (Cite Lesson 00022)
MONTAGE_SIZE = SLICE_SIZE * GRID_SIZE  # 224x224 input to the model

# Slice selection
# We select the middle slice (Cite Lesson 00015)
SLICE_DEPTHS = [0.5]

# Modalities to use (mapped to RGB channels)
# Channel 0: FLAIR
# Channel 1: T1wCE
# Channel 2: T2w
SELECTED_MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# ==========================================
# Model Configuration
# ==========================================
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1
PRETRAINED = True
DROPOUT_RATE = 0.2  # Default for EfficientNet-B0

# ==========================================
# Training Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 32  # Increased for stability (Cite Lesson 00008)
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
N_FOLDS = 5
NUM_WORKERS = 4

# Early Stopping
PATIENCE = 3
MIN_DELTA = 0.001

# ==========================================
# Caching Configuration
# ==========================================
# Filenames for cached datasets
CACHE_TRAIN_NAME = "cached_montage_train.parquet"
CACHE_VAL_NAME = "cached_montage_val.parquet"
CACHE_TEST_NAME = "cached_montage_test.parquet"
