import os
import torch

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
N_JOBS = 12  # Utilizing available vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False  # Set to True for quick debugging runs
DEBUG_SAMPLE_SIZE = 100 if DEBUG else None

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_37"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATASET COLUMNS
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text Columns for NLP
TEXT_COLS = {
    "title": "request_title",
    "body": "request_text_edit_aware",  # Using edit-aware text to prevent leakage
    "subreddits": "requester_subreddits_at_request",
}

# Columns to exclude from raw metadata features (ID, Target, Text, Leakage)
METADATA_EXCLUDE_COLS = [
    "request_id",
    "requester_received_pizza",
    "request_text",
    "request_text_edit_aware",
    "request_title",
    "requester_subreddits_at_request",
    "giver_username_if_known",
    "source_file",
    "requester_username",
    "requester_user_flair",  # High missing rate / categorical
    "post_was_edited",  # Potential leakage if not handled carefully
]

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================
# Text Processing
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_TEXT_LENGTH = 512

# TF-IDF Settings (Stream A)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Community Profiling
TOP_K_SUBREDDITS = 50  # Number of top subreddits to use as binary indicators

# History Sequence Settings (Stream B)
MAX_HISTORY_LENGTH = 50  # Max number of past subreddits to consider in sequence

# =============================================================================
# MODEL HYPERPARAMETERS: STREAM A (RANDOM FOREST)
# =============================================================================
# Dispersion-Normalized Random Forest
RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 1,  # Low regularization to capture fine signals
    "class_weight": "balanced",  # Handle moderate imbalance
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
    "verbose": 0,
}

# =============================================================================
# MODEL HYPERPARAMETERS: STREAM B (MLP)
# =============================================================================
# Centroid-Augmented Dual-Attention MLP
MLP_PARAMS = {
    # Architecture
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dims": [256, 128],  # Hidden layers after fusion
    "attention_dim": 64,  # Dimension for attention projection
    # Regularization (Dropout Only regime)
    "dropout_rate": 0.2,  # Dropout for dense layers
    "embedding_dropout": 0.5,  # Dropout for input embeddings
    "use_batch_norm": False,  # Explicitly disabled per Idea description
    # Training
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,  # AdamW weight decay
    "epochs": 50,
    "patience": 15,  # Early stopping patience
    "grad_clip": 1.0,  # Gradient clipping
}

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
