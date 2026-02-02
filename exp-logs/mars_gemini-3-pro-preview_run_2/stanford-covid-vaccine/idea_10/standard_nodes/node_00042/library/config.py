import os
import torch

# =============================================================================
# FILE PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_10"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Path for the final submission file
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Fixed length of RNA sequences in the dataset
SEQ_LEN = 107

# The number of positions scored in the public/private test sets (usually 68)
# However, we predict for the full length (107) and mask during scoring.
PRED_LEN = 107

# Columns required for the competition metric (MCRMSE)
# We only compute loss on these columns to avoid negative transfer from auxiliary targets.
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

# All available target columns in the training data
ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

# Number of output channels
NUM_TARGETS = len(ALL_TARGETS)

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
# Vocab sizes for One-Hot Encoding
VOCAB_SIZE_SEQ = 4  # A, G, U, C
VOCAB_SIZE_STRUCT = 3  # (, ), .
VOCAB_SIZE_LOOP = 7  # S, M, I, B, H, E, X

# Total input channels for the model
# Note: We explicitly exclude 'Partner Identity' to force learning from interaction context.
INPUT_CHANNELS = VOCAB_SIZE_SEQ + VOCAB_SIZE_STRUCT + VOCAB_SIZE_LOOP

# =============================================================================
# CACHE MANAGEMENT
# =============================================================================
# Unique identifier for this data processing version.
# Changing this string forces the data loader to re-process raw data instead of loading
# from .npy/.parquet files, ensuring the new feature set is used.
CACHE_VERSION = "idea_10_staged_dense_v1"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Staged Interactive Dense Network Configuration

# Dimension of the latent feature space
HIDDEN_DIM = 128

# Convolutional Kernel Size
KERNEL_SIZE = 3

# Dropout rate applied within blocks
DROPOUT = 0.1

# Dilation schedule for the TCN blocks.
# This list defines the depth and receptive field of ONE stage.
# The model will have two such stages (Local Context & Pair Context).
DILATIONS = [1, 2, 4, 8, 16, 32]

# BiGRU Configuration (Global Aggregation)
# Output of BiGRU will be concatenated (hidden_dim // 2 * 2 = hidden_dim)
GRU_HIDDEN_DIM = HIDDEN_DIM // 2
GRU_LAYERS = 1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 50
PATIENCE = 10  # Early stopping patience

# Random Seed for reproducibility
SEED = 42

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
