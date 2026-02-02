import os
import torch

# ====================================================
# Directory and File Paths
# ====================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_27"

# Ensure the working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ====================================================
# Global Constants & Reproducibility
# ====================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================
# Data Configuration (SIRV Strategy)
# ====================================================
IMG_SIZE = 224
# Input channels = 3 modalities * 3 depths = 9 channels
MODALITIES = ["FLAIR", "T1wCE", "T2w"]
RELATIVE_DEPTHS = [0.4, 0.5, 0.6]

# ====================================================
# Training Hyperparameters
# ====================================================
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# Regularization
WEIGHT_DECAY = 1e-2  # Aggressive decay as per strategy
DROPOUT_RATE = 0.3

# ====================================================
# Debugging & Development Control
# ====================================================
# If set to an integer (e.g., 50), only that many samples will be used for training/testing.
# Set to None for full run.
DEBUG_DATA_LIMIT = None
