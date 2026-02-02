import os
import torch

# =============================================================================
# Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
OUTPUT_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# File Paths
# =============================================================================
# Using metadata files as the source of truth for splits
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Path for the final submission file
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Model Hyperparameters
# =============================================================================
MAX_LEN_DEEP = 200  # Sequence length for the Deep component
VOCAB_SIZE_DEEP = 50000  # Vocabulary size for the Deep component (Embedding)
VOCAB_SIZE_WIDE = 100000  # Vocabulary size for the Wide component (TF-IDF)
EMBED_DIM = 128  # Dimension of embeddings in Deep component
NUM_TAGS = 5000  # Number of top frequent tags to predict

# =============================================================================
# Training Settings
# =============================================================================
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3  # Stop if validation metric doesn't improve for 3 epochs

# =============================================================================
# Hardware & Reproducibility
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of subprocesses for data loading
SEED = 42  # Fixed random seed for reproducibility
