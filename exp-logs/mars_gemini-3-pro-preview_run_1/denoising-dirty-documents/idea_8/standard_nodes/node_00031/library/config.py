import os
import torch

# =============================================================================
# Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for the specific solution iteration (Idea 8)
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# File Paths
# =============================================================================
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Data Configuration
# =============================================================================
# Optimal patch size balancing context and crop diversity (Lesson 27)
PATCH_SIZE = 160
BATCH_SIZE = 16
# Number of data loading workers (12 vCPUs available)
NUM_WORKERS = 4

# =============================================================================
# Model Configuration
# =============================================================================
# 3-Level Encoder-Decoder structure (32 -> 64 -> 128 filters) as per Lesson 18
MODEL_CHANNELS = [32, 64, 128]
# ASPP dilations for the bottleneck to increase receptive field without downsampling
ASPP_DILATIONS = [1, 2, 4, 8]

# =============================================================================
# Training Configuration
# =============================================================================
# Full convergence strategy (Lesson 12 & 16)
EPOCHS = 1000
LEARNING_RATE = 1e-3
# 5 independent seeds for fully converged seed-averaging ensemble
SEEDS = [42, 43, 44, 45, 46]

# =============================================================================
# Compute Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Inference Configuration
# =============================================================================
# D4 Group Test-Time Augmentation (8 views) as per Lesson 28
TTA_VIEWS = 8
