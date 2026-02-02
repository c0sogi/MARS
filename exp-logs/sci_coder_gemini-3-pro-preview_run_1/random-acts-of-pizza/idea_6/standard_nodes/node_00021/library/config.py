import os
import torch

# ==========================================
# Path Configuration
# ==========================================
# Base directories
INPUT_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Data Paths (Metadata CSVs)
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
VAL_PATH = os.path.join(INPUT_DIR, "val.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

# Output Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths for Deterministic Processing
# We use Parquet for DataFrames and NPY for embeddings
TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

# Cache for SBERT Embeddings (dense numpy arrays)
TRAIN_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "train_embeddings.npy")
VAL_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "val_embeddings.npy")
TEST_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "test_embeddings.npy")

# ==========================================
# Data Configuration
# ==========================================
RANDOM_STATE = 42
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"
TEXT_COL = "request_text_edit_aware"

# Leakage Prevention:
# Columns containing these substrings represent future information (at retrieval)
# or target leakage, and must be excluded from features.
LEAKAGE_KEYWORDS = [
    "at_retrieval",  # Data collected long after request
    "giver_username",  # Only present if successful
    "post_was_edited",  # Often indicates "Thanks for pizza" edit
    "requester_user_flair",  # Flair updates upon success (shroom/PIF)
    "timestamp",  # Proxy for retrieval time/split
]

# Specific columns to drop that might not be caught by keywords
DROP_COLS = ["source_file", "request_text"]  # We use the edit_aware version

# ==========================================
# Model Configuration
# ==========================================

# Learner A: Lexical-Ratio Random Forest
RF_CONFIG = {
    "n_estimators": 500,  # High number for stability
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",  # Handle moderate imbalance
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 0,
}

# Learner B: Stabilized Dual-Branch MLP
MLP_CONFIG = {
    # Architecture Dimensions
    "embedding_dim": 384,  # SBERT output dimension
    "semantic_hidden_dim": 128,  # Branch 1 (Text) hidden size
    "ratio_hidden_dim": 64,  # Branch 2 (Tabular) hidden size
    "fusion_hidden_dim": 64,  # Fusion layer size
    "output_dim": 1,
    # Regularization
    "semantic_dropout": 0.5,  # High dropout for text branch to prevent overfitting
    "ratio_dropout": 0.1,  # Low dropout for engineered ratios
    # Training Hyperparameters
    "batch_size": 32,
    "epochs": 50,  # Extended training for convergence
    "patience": 15,  # High patience to survive early fluctuations
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,  # AdamW regularization
    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==========================================
# Ensemble Configuration
# ==========================================
# Simple Weighted Average Strategy
ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
