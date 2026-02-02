import os
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Homogeneous Seed Averaging Strategy
NUM_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]
MASTER_SEED = 42

# =============================================================================
# COMPUTE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of data loading workers
NUM_WORKERS = 2

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Custom Narrow SE-ResNet with Selective Texture-Context Aggregation
INPUT_SHAPE = (3, 32, 32)  # (Channels, Height, Width)
NUM_CLASSES = 1

# Channel configuration for the narrow backbone
# Stage 1 -> Stage 2 (Texture/GCP) -> Stage 3 (Context/GAP)
CHANNEL_CONFIG = [16, 32, 64]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 128
EPOCHS = 15

# Optimizer (AdamW)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

# Scheduler (Cosine Annealing)
T_MAX = EPOCHS
ETA_MIN = 1e-6

# Early Stopping
PATIENCE = 5
MIN_DELTA = 1e-4

# =============================================================================
# INFERENCE
# =============================================================================
# Test Time Augmentation settings
USE_TTA = True
