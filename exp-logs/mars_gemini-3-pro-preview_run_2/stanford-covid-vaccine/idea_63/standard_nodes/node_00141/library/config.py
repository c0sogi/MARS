import os
import random
import numpy as np
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = "./working/idea_63"

# Ensure the specific cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
SEQ_LENGTH = 107
SCORED_LENGTH = 68

# Target Columns
# All 5 ground truth columns provided in training data
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
# The 3 columns actually used for the MCRMSE score
SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

NUM_TARGETS = len(TARGET_COLS)  # 5

# Debugging / Development
DEBUG = False
DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

# =============================================================================
# MODEL HYPERPARAMETERS (HS-GFDN)
# =============================================================================
# Main Backbone
GROWTH_RATE = 64
LATENT_DIM = 64
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]
DROPOUT = 0.1

# Feedback Stem & Backbone
FEEDBACK_GROWTH_RATE = 16
FEEDBACK_CHANNELS = 32

# Aggregation
RNN_HIDDEN_SIZE = 64

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Optimization Strategy: Small Batch Regime
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Training Loop
EPOCHS = 50
PATIENCE = 7  # Early stopping patience

# Reproducibility
SEED = 42


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
