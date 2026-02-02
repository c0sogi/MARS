import os
import torch

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_9"
SUBMISSION_DIR = "./submission"

# Ensure mutable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Directory for deterministic data processing
CACHE_DIR = WORKING_DIR

# =============================================================================
# Model & Data Configuration
# =============================================================================
NUM_CLASSES = 19
IMG_HEIGHT = 256
CHANNELS = 3  # RGB (Replicated)

# Multi-Resolution Ensemble Configuration
# Teacher models trained at different widths to capture different features
# Resolution A (Dense): 384
# Resolution B (Balanced): 512
# Resolution C (Detailed): 640
TEACHER_WIDTHS = [384, 512, 640]

# Student Model Configuration
# The student operates at the balanced resolution
STUDENT_WIDTH = 512

# =============================================================================
# Training Hyperparameters
# =============================================================================
SEED = 42
BATCH_SIZE = 16  # Optimized for small dataset stability
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Epochs
TEACHER_EPOCHS = 30
STUDENT_EPOCHS = 50

# SWA (Stochastic Weight Averaging) Configuration
# Activate SWA for the last 30% of student training
SWA_START_EPOCH = int(STUDENT_EPOCHS * 0.7)
SWA_LR = 1e-4

# Augmentation
MIXUP_ALPHA = 0.2

# =============================================================================
# Compute & Debugging
# =============================================================================
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debugging flags to control dataset size for rapid testing
DEBUG = False
DEBUG_SUBSET_SIZE = 20
