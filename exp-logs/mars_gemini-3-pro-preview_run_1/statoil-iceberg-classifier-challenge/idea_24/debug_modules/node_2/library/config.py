import os
import torch

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_24"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# ==========================================
# Global Statistics (Derived from Training Data)
# ==========================================
# Used for 'Normalize then Average' logic.
# Values taken from Data Analysis:
# Band 1 (HH) - Min: -45.5944, Max: 32.1806
# Band 2 (HV) - Min: -45.6555, Max: 17.8628
BAND1_MIN = -45.5944
BAND1_MAX = 32.1806
BAND2_MIN = -45.6555
BAND2_MAX = 17.8628

# ==========================================
# Data Parameters
# ==========================================
ORIGINAL_IMG_SIZE = 75
IMG_SIZE = 224  # Upsampled size for ResNet
BATCH_SIZE = 32
NUM_WORKERS = 2  # Adjusted for vCPU availability

# ==========================================
# Model Parameters
# ==========================================
MODEL_NAME = "resnet18"
PRETRAINED = True
NUM_CLASSES = 1
DROPOUT_RATE = 0.5

# ==========================================
# Training Hyperparameters
# ==========================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimizer (SAM + AdamW)
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
SAM_RHO = 0.05

# Scheduler (ReduceLROnPlateau)
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_MODE = "min"

# Loss
LABEL_SMOOTHING = 0.05

# Phase 1: Calibration
MAX_EPOCHS_PHASE_1 = 100  # Upper bound, relies on Early Stopping
EARLY_STOPPING_PATIENCE = 15  # Allow enough time to overcome plateaus

# Phase 2: Production (SWA)
# SWA starts immediately after the calibrated epoch count
SWA_DURATION_EPOCHS = 12
SWA_LR = 1e-4  # Usually slightly lower or same as final LR, but fixed for SWA

# ==========================================
# TTA Parameters
# ==========================================
# Klein Four-Group: Original, H-Flip, V-Flip, Rotate180
TTA_STEPS = 4
