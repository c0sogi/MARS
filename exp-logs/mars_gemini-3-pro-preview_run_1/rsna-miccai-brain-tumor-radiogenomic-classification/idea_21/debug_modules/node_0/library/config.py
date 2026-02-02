import os
import torch

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata Paths
TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output & Cache Directories
WORKING_DIR = "./working/idea_21"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Create necessary directories immediately
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Processing Configuration
# ==========================================
IMAGE_SIZE = 224
STRIDE = 5
NUM_CHANNELS = 9  # 3 modalities (FLAIR, T1wCE, T2w) * 3 depths (M-s, M, M+s)

# Modalities used in the 9-channel stack
# Order matters for the weight inflation logic
MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# ==========================================
# Model & Training Hyperparameters
# ==========================================
# Backbone
MODEL_NAME = "efficientnet_b0"

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.3
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# ==========================================
# Hardware & Reproducibility
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
SEED = 42

# ==========================================
# Debugging
# ==========================================
# If True, runs on a small subset of data for testing pipeline flow
DEBUG = False
