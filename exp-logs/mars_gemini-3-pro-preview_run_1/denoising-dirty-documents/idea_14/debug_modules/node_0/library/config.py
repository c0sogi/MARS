import os
import torch

# -----------------------------------------------------------------------------
# Directory Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory specific to this strategy (Idea 14)
WORKING_DIR = "./working/idea_14"
SUBMISSION_DIR = "./submission"

# Ensure necessary output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# File Path Configuration
# -----------------------------------------------------------------------------
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Path to the sample submission file provided in the input
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

# Path for the final generated submission
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Hyperparameters
# -----------------------------------------------------------------------------
# Patch size for random crops during training (160x160)
PATCH_SIZE = 160

# Batch size (Small batch size of 16)
BATCH_SIZE = 16

# Number of CPU workers for data loading (12 vCPUs available)
NUM_WORKERS = 12

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
# Learning rate for Adam optimizer
LEARNING_RATE = 1e-3

# Total number of epochs for full convergence
NUM_EPOCHS = 1000

# Random seeds for the 10-model ensemble
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Input and Output channels (Grayscale images)
IN_CHANNELS = 1
OUT_CHANNELS = 1

# U-Net Depth (3 levels of downsampling: 32 -> 64 -> 128)
DEPTH = 3
START_FILTERS = 32

# -----------------------------------------------------------------------------
# Compute Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_checkpoint_path(seed):
    """
    Returns the file path for saving/loading the model checkpoint corresponding to a specific seed.
    """
    return os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")


def get_cache_path(split_name):
    """
    Returns the file path for the cached dataset (npz format).
    split_name should be 'train', 'val', or 'test'.
    """
    return os.path.join(WORKING_DIR, f"{split_name}_cache.npz")
