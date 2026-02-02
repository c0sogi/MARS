import os
import torch

# ==========================================
# Path Configurations
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Paths
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Working and Output Directories
WORKING_DIR = "./working/idea_7"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Path
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Hyperparameters
# ==========================================
# Idea: High-Density Uniform Sampling (32 slices per modality)
SLICES_PER_MODALITY = 32
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
NUM_MODALITIES = len(MODALITIES)

# Total input channels = 32 slices * 4 modalities = 128
IN_CHANNELS = SLICES_PER_MODALITY * NUM_MODALITIES

# Image Spatial Dimensions
IMG_SIZE = 256

# ==========================================
# Model Hyperparameters
# ==========================================
# Channel-Attention Stem settings
STEM_REDUCTION_RATIO = 8  # For SE Block in the stem
BOTTLENECK_CHANNELS = 64  # Compress 128 -> 64 before backbone

# Backbone
BACKBONE_NAME = "efficientnet_b0"

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15

# Optimizer settings
WEIGHT_DECAY = 0.0  # Explicitly set to 0 as per Idea (Adam, no WD)

# Compute settings
NUM_WORKERS = 4  # Optimized for 12 vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Reproducibility & Debugging
# ==========================================
SEED = 42
DEBUG = False
DEBUG_SAMPLE_SIZE = 20  # Number of samples to use if DEBUG is True
