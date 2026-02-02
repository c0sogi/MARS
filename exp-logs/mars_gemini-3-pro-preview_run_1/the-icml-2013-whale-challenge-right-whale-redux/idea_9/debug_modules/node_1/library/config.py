import os
import torch

# =============================================================================
# File Paths and Directories
# =============================================================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for the current experiment (Idea 9)
WORKING_DIR = "./working/idea_9"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# Audio Processing Hyperparameters
# =============================================================================
SAMPLE_RATE = 2000
N_MELS = 128
HOP_LENGTH = 10
N_FFT = 512  # FFT window size, typically larger than hop_length.
# 512 samples @ 2000Hz is ~256ms, covering the low freq features.

# =============================================================================
# Model Architecture Hyperparameters
# =============================================================================
NUM_CLASSES = 1
MODEL_BACKBONE = "skresnet18"  # Selective Kernel ResNet-18 from timm
USE_PRETRAINED = True
GRU_HIDDEN_SIZE = 256
GRU_LAYERS = 2

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
POS_WEIGHT = 9.0  # Weight for positive class (whale call) to handle imbalance
MIXUP_ALPHA = 0.4  # Alpha parameter for Mixup augmentation

# =============================================================================
# Runtime and Reproducibility
# =============================================================================
SEED = 42
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debugging parameters
DEBUG = False
DEBUG_SIZE = 100  # Subset size when DEBUG is True
