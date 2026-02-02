import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORK_DIR = "./working/idea_34"
SUBMISSION_DIR = "./submission"

# Ensure necessary output directories exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
# Image Dimensions
IMG_SIZE = 224

# Modalities to use (Order matters for channel stacking)
# The strategy uses these three modalities to form the RGB-like structure per depth
MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# Relative depths for ROI sampling (Scale-Invariant)
# 0.4 = 40% depth, 0.5 = Center (50%), 0.6 = 60% depth
ROI_RELATIVE_DEPTHS = [0.4, 0.5, 0.6]

# Total input channels = len(MODALITIES) * len(ROI_RELATIVE_DEPTHS)
# 3 modalities * 3 depths = 9 channels
NUM_CHANNELS = len(MODALITIES) * len(ROI_RELATIVE_DEPTHS)

# ==========================================
# Model Configuration
# ==========================================
BACKBONE = "efficientnet_b0"
PRETRAINED = True
NUM_CLASSES = 1
DROPOUT_RATE = 0.3  # Enforced dropout rate as per strategy

# ==========================================
# Training Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay to prevent overfitting
EARLY_STOPPING_PATIENCE = 5
N_FOLDS = 5

# ==========================================
# Hardware & Execution
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of subprocesses for data loading
