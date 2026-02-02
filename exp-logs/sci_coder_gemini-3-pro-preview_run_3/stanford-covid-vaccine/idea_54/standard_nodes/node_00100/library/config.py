import os
import torch
import numpy as np
import random

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_54")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths (Parquet files)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Submission Paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
SEQ_LEN = 107
PRED_LEN = 68

# Input Features:
# 4 Nucleotides (A, G, C, U)
# 3 Structure types ( (, ), . )
# 7 Predicted Loop types (S, M, I, B, H, E, X)
INPUT_DIM = 4 + 3 + 7  # 14

# Targets
OUTPUT_DIM = 5
ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

# =============================================================================
# MODEL ARCHITECTURE (Deep Stabilized Bias-Refined Decoupled BiGRU)
# =============================================================================
HIDDEN_DIM = 384
NUM_LAYERS = 4
KERNEL_SIZE = 3
DROPOUT = 0.1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10  # For Early Stopping

# Gradient Clipping is mandatory for the 4-layer hybrid architecture
GRAD_CLIP = 1.0

# =============================================================================
# DEBUGGING AND REPRODUCIBILITY
# =============================================================================
# Set to True to run on a small subset of data for testing pipeline
DEBUG = False
DEBUG_SUBSET_SIZE = 50

SEED = 2024


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
