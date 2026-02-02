import os
import torch
import random
import numpy as np

# ==========================================
# Global Configuration Module
# ==========================================

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42


def seed_everything(seed=SEED):
    """
    Sets the random seed for all relevant libraries to ensure deterministic behavior.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Data Paths & Directories
# ---------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Files (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Working Directory for Caching Intermediate Data
# Using 'idea_23' as the designated workspace for this run
WORKING_DIR = "./working/idea_23"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ---------------------------
# Data Processing Hyperparameters
# ---------------------------
IMG_SIZE = 224

# RMS-HD Network Specifics:
# We use 32 slices per modality to capture the tumor in the Z-axis.
NUM_SLICES_PER_MODALITY = 32

# Strict Modality Ordering for Semantic Stability
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
NUM_MODALITIES = len(MODALITIES)

# Total Input Channels = 32 slices * 4 modalities = 128
TOTAL_INPUT_CHANNELS = NUM_SLICES_PER_MODALITY * NUM_MODALITIES

# ---------------------------
# Model Architecture
# ---------------------------
BACKBONE_NAME = "efficientnet_b0"
STEM_OUT_CHANNELS = 64  # Compressing 128 channels to 64 in the stem
DROP_PATH_RATE = 0.2  # Regularization for the backbone

# ---------------------------
# Training Hyperparameters
# ---------------------------
# Batch size 16 fits within A100 memory for this high-channel input
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
PATIENCE = 3  # Early stopping patience

# ---------------------------
# Hardware
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
