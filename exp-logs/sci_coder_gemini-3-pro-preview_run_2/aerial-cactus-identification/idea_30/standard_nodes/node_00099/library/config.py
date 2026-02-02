import os
import torch

# -----------------------------------------------------------------------------
# Directories and Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Output directory specifically for this idea/experiment
OUTPUT_DIR = "./working/idea_30"

# Ensure the output directory exists immediately upon config import
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metadata File Paths
# These files contain the stratified splits and file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 32
NUM_CLASSES = 1
NUM_WORKERS = 4  # Optimized for the available vCPUs

# -----------------------------------------------------------------------------
# Model Configuration
# Wide ResNeXt-ECA with Multi-Scale GAP Aggregation
# -----------------------------------------------------------------------------
MODEL_NAME = "WideResNeXt_MultiScale_ECA"

# "Super-Wide" Channel Configuration as per solution_lesson_node_00064
CHANNELS = [64, 128, 256]
CARDINALITY = 32

# Structural Innovations
USE_ECA = True

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Homogeneous Seed Averaging (5 independent instances)
SEEDS = [0, 1, 2, 3, 4]

# -----------------------------------------------------------------------------
# Evaluation / Inference Configuration
# -----------------------------------------------------------------------------
# Test Time Augmentation: Original + Horizontal Flip + Vertical Flip
USE_TTA = True

# -----------------------------------------------------------------------------
# Hardware
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
