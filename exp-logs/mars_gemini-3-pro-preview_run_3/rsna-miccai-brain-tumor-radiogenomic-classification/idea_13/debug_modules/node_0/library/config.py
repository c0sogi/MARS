import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"

# Metadata Parquet Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Working & Output Directories
# idea_13 corresponds to the current experiment iteration
CACHE_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")

# Ensure necessary writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Hyperparameters
# ==========================================
IMG_SIZE = 256
NUM_SLICES_PER_MODALITY = 32
NUM_MODALITIES = 4

# The model takes a stack of slices from all modalities
# 32 slices * 4 modalities = 128 input channels
INPUT_CHANNELS = NUM_SLICES_PER_MODALITY * NUM_MODALITIES

# ==========================================
# Model Hyperparameters
# ==========================================
BACKBONE_NAME = "efficientnet_b0"
# Groups must match NUM_SLICES_PER_MODALITY for the interleaved strategy
# to ensure each group processes exactly one slice depth (4 channels)
STEM_GROUPS = 32
STEM_OUT_CHANNELS = 128
COMPRESSED_CHANNELS = 64

# ==========================================
# Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 15
PATIENCE = 5  # Early stopping patience

# ==========================================
# Compute & Debugging
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# Set to an integer (e.g., 50) to limit dataset size for fast debugging/verification
# Set to None for full training run
MAX_SAMPLES = None
