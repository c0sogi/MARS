import os
import torch

# -----------------------------------------------------------------------------
# Global System Configurations
# -----------------------------------------------------------------------------
SEED = 42
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# -----------------------------------------------------------------------------
# Data Processing Configuration
# -----------------------------------------------------------------------------
# Debugging: Set to True to train/validate on a small subset of data
DEBUG = False
DEBUG_SAMPLE_SIZE = 2000

# Caching: Parquet files for processed datasets
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

# -----------------------------------------------------------------------------
# Sparse Stream (Ridge Regression) Hyperparameters
# -----------------------------------------------------------------------------
# TF-IDF Vectorizer Settings
VOCAB_SIZE = 60000
NGRAM_RANGE = (1, 2)  # Bigrams
USE_IDF = True
SUBLINEAR_TF = True  # Logarithmic Term Frequency
STRIP_ACCENTS = None  # Preserve accents for technical precision
BINARY = False

# Ridge Regression Settings
RIDGE_SOLVER = "lsqr"  # Efficient solver for large sparse matrices
RIDGE_TOL = 1e-4

# Artifact Paths
VECTORIZER_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
RIDGE_MODEL_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")

# -----------------------------------------------------------------------------
# Dense Stream (Transformer) Hyperparameters
# -----------------------------------------------------------------------------
MODEL_NAME = "microsoft/codebert-base"

# Input Formatting
MAX_LEN = 256  # Max sequence length for tokenization
ANCHOR_CHAR_LIMIT = 128  # Characters to extract from Start/End code cells
TOP_K_KEYWORDS = 10  # Number of TF-IDF keywords for Topic Anchors

# Training Hyperparameters
BATCH_SIZE = 32
LR = 2e-5
NUM_EPOCHS = 1  # Limited to 1 epoch for runtime constraints
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0

# Artifact Path
TRANSFORMER_MODEL_PATH = os.path.join(WORKING_DIR, "transformer_model.pth")

# -----------------------------------------------------------------------------
# Ensemble Configuration
# -----------------------------------------------------------------------------
# Weighting factor for the ensemble
# Final Rank = ALPHA * Ridge_Rank + (1 - ALPHA) * Transformer_Rank
ALPHA = 0.6
