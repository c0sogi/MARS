import os
import random
import numpy as np
import torch

# =============================================================================
# FILE PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
SEQ_LENGTH = 107
SEQ_SCORED = 68

# Target columns for training and prediction
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
NUM_TARGETS = len(TARGET_COLS)

# Vocabulary Mappings
# Sequence: A, G, U, C
TOKEN2INT_SEQ = {"A": 0, "G": 1, "U": 2, "C": 3}
VOCAB_SIZE_SEQ = len(TOKEN2INT_SEQ)

# Structure: (, ), .
TOKEN2INT_STRUCT = {".": 0, "(": 1, ")": 2}
VOCAB_SIZE_STRUCT = len(TOKEN2INT_STRUCT)

# Predicted Loop Type: B, E, H, I, M, S, X
TOKEN2INT_LOOP = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}
VOCAB_SIZE_LOOP = len(TOKEN2INT_LOOP)

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Embedding dimensions for each input feature
EMBED_DIM_SEQ = 32
EMBED_DIM_STRUCT = 16
EMBED_DIM_LOOP = 16

# Total input dimension to the encoder (sum of embeddings)
INPUT_DIM = EMBED_DIM_SEQ + EMBED_DIM_STRUCT + EMBED_DIM_LOOP

# Bi-GRU Encoder settings
HIDDEN_DIM_GRU = 256
NUM_LAYERS_GRU = 3
DROPOUT_GRU = 0.3

# Transformer Encoder settings
# Note: d_model for transformer will be 2 * HIDDEN_DIM_GRU (bidirectional output)
NUM_LAYERS_TRANSFORMER = 2
NHEAD_TRANSFORMER = 8
DROPOUT_TRANSFORMER = 0.1
DIM_FEEDFORWARD = 1024

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
PATIENCE = 7  # For Early Stopping

# Debugging / Development
DEBUG = False
DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the seed immediately upon import
seed_everything(SEED)
