import os
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_27"
SUBMISSION_DIR = "./submission"

# Data file paths (using generated metadata)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission output path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache directories for intermediate artifacts
CACHE_DIR = WORKING_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# Debugging: Set to a small integer (e.g., 100) to limit dataset size for testing.
# Set to None for full training.
DATA_SAMPLE_SIZE = None

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================
# Text Processing
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# TF-IDF (for Random Forest)
TFIDF_VOCAB_SIZE = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Community History
TOP_K_SUBREDDITS = 50  # Number of top subreddits to use as binary indicators

# =============================================================================
# STREAM A: RANDOM FOREST HYPERPARAMETERS
# =============================================================================
# Configuration designed for Low-Bias to preserve sparse signals
RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 1,  # Minimal regularization
    "max_depth": None,  # Allow full depth
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "verbose": 0,
}

# =============================================================================
# STREAM B: DUAL-QUERY MLP HYPERPARAMETERS
# =============================================================================
# Training settings
MLP_BATCH_SIZE = 32
MLP_EPOCHS = 50
MLP_PATIENCE = 15  # High patience for stabilization
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 1e-4

# Architecture settings
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.3

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Simple Weighted Average
WEIGHT_RF = 0.5
WEIGHT_MLP = 0.5
