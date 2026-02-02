import os

# =============================================================================
# Global Configuration
# =============================================================================
RANDOM_STATE = 42

# =============================================================================
# File Paths
# =============================================================================
# Input Metadata (Pre-split CSVs)
TRAIN_DATA_PATH = "./metadata/train.csv"
VAL_DATA_PATH = "./metadata/val.csv"
TEST_DATA_PATH = "./metadata/test.csv"

# Working Directory for Caching Intermediate Artifacts
# Stores processed tensors, embeddings, and matrices
CACHE_DIR = "./working/idea_5/"

# Submission Output
SUBMISSION_DIR = "./submission/"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Data Column Definitions
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"
TEXT_COL = "request_text_edit_aware"

# =============================================================================
# Leakage Prevention & Feature Selection
# =============================================================================
# Suffixes indicating data available only after the request (Leakage)
LEAKAGE_SUFFIXES = ["_at_retrieval"]

# Specific columns to exclude manually (Leakage, IDs, or Raw Text variants)
EXCLUDED_COLS = [
    "giver_username_if_known",
    "request_text",  # We use 'request_text_edit_aware'
    "unix_timestamp_of_request_utc",  # Redundant with 'unix_timestamp_of_request'
    "source_file",  # Metadata artifact
    "request_title",  # Often redundant or handled via text concatenation if needed
]

# =============================================================================
# Feature Engineering Hyperparameters
# =============================================================================
# Stream A: TF-IDF Vectorization
TFIDF_PARAMS = {
    "max_features": 3000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "binary": True,
    "sublinear_tf": True,
}

# Stream B: Sentence Transformer
# Pre-trained model for generating dense semantic embeddings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# =============================================================================
# Model Hyperparameters
# =============================================================================

# Stream A: Random Forest Classifier
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Stream B: Dual-Branch MLP
MLP_PARAMS = {
    # Architecture
    "semantic_input_dim": EMBEDDING_DIM,
    "hidden_dim": 128,
    "dropout_rate": 0.5,
    # Training
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "epochs": 50,
    "patience": 8,  # Early stopping patience
    "device": "cuda",  # Logic should handle fallback to cpu
}

# =============================================================================
# Ensemble Configuration
# =============================================================================
# Weights for weighted average ensemble
ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
