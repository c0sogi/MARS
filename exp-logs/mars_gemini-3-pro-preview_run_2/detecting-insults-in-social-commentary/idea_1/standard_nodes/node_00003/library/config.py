import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission_null.csv")

# ==========================================
# Data Configuration
# ==========================================
MAX_VOCAB_SIZE = 10000
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
MIN_FREQ = 2  # Minimum frequency for a word to be included in the vocabulary

# ==========================================
# Model Configuration
# ==========================================
EMBED_DIM = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 1

# ==========================================
# Training Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2
WEIGHT_DECAY = 1e-5  # L2 Regularization to prevent overfitting

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2  # Number of subprocesses for data loading
