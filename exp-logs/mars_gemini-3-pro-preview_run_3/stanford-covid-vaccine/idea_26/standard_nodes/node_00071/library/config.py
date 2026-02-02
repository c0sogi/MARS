import os
import random
import numpy as np
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_26"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
SEQ_LEN = 107
SEQ_SCORED = 68

# Vocabulary Sizes for One-Hot Encoding
# Sequence: A, G, C, U
VOCAB_SIZE_SEQ = 4
# Structure: (, ), .
VOCAB_SIZE_STRUCT = 3
# Predicted Loop Type: S, M, I, B, H, E, X
VOCAB_SIZE_LOOP = 7

# Total Input Channels (4 + 3 + 7 = 14)
INPUT_CHANNELS = VOCAB_SIZE_SEQ + VOCAB_SIZE_STRUCT + VOCAB_SIZE_LOOP

# Column Definitions
ID_COL = "id"
SEQUENCE_COL = "sequence"
STRUCTURE_COL = "structure"
LOOP_TYPE_COL = "predicted_loop_type"
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_TARGET_INDICES = [0, 1, 3]
NUM_TARGETS = len(TARGET_COLS)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Architecture: Deep Post-Norm BiGRU with Zero-Masked Channel-Gating
HIDDEN_DIM = 384
NUM_LAYERS = 4
DROPOUT = 0.1

# Convolutional Stem Configuration
CONV_FILTERS = 256
CONV_KERNEL_SIZE = 3

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 5  # Early stopping patience
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold (Critical for stability)

# Debugging / Subset Control
DEBUG = False
DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42


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
