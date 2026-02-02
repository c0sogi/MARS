import os
import torch
import numpy as np
import random

# =============================================================================
# DIRECTORY & FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"

# Ensure working directory exists for cache and outputs
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
CACHE_DIR = WORKING_DIR
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission/submission.csv"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
SEQ_LENGTH = 107
SEQ_SCORED = 68

# Column Names
ID_COL = "id"
SEQUENCE_COL = "sequence"
STRUCTURE_COL = "structure"
LOOP_TYPE_COL = "predicted_loop_type"

# Target Columns
# We train on all 5 conditions to learn shared physics
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
# Only these 3 are used for the competition metric
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SCORED_INDICES = [i for i, x in enumerate(TARGET_COLS) if x in SCORED_TARGETS]

# Vocabularies / Mappings
TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGCU")}
TOKEN2INT_STRUCT = {x: i for i, x in enumerate("().")}
TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}

# Feature Dimensions
NUM_SEQ_TOKENS = 4  # A, G, C, U
NUM_STRUCT_TOKENS = 3  # (, ), .
NUM_LOOP_TOKENS = 7  # S, M, I, B, H, E, X
NUM_PARTNER_TOKENS = 4  # Partner base identity (A, G, C, U) or zeros

# Total Input Channels: 4 + 3 + 7 + 4 = 18
INPUT_CHANNELS = (
    NUM_SEQ_TOKENS + NUM_STRUCT_TOKENS + NUM_LOOP_TOKENS + NUM_PARTNER_TOKENS
)

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# 1D CNN Backbone (Dilated Residual TCN)
CNN_CHANNELS = 128
KERNEL_SIZE = 3
# Exponential dilation rates to cover global context without pooling
DILATIONS = [1, 2, 4, 8, 16, 32]
DROPOUT = 0.2

# RNN Global Aggregation (BiGRU)
# Hidden size is set to half of CNN channels so bidirectional output matches CNN output dim
RNN_HIDDEN_SIZE = CNN_CHANNELS // 2
RNN_LAYERS = 1
RNN_BIDIRECTIONAL = True

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debugging flags
DEBUG = False
DEBUG_SUBSET_SIZE = 100


# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed=SEED):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
