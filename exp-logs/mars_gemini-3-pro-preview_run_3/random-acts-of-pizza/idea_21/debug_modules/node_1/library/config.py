import os

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
# Metadata directories containing the stratified train/val and test splits
METADATA_DIR = "./metadata"
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Directory for caching intermediate processed features (embeddings, sparse matrices)
CACHE_DIR = "./working/idea_21/"
os.makedirs(CACHE_DIR, exist_ok=True)

# Directory for final submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Column Configuration
# -----------------------------------------------------------------------------
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text input: Use the edit-aware version to avoid leakage from "EDIT: Thanks!"
TEXT_COL = "request_text_edit_aware"
TITLE_COL = "request_title"

# Behavioral input: Space-separated list of subreddits
HISTORY_COL = "requester_subreddits_at_request"

# Source file tracking column
SOURCE_COL = "source_file"

# Columns to strictly exclude from feature sets
# We exclude IDs, target, raw text (redundant), and potential leakage or high-cardinality columns
EXCLUDE_COLS = [
    ID_COL,
    TARGET_COL,
    TEXT_COL,
    TITLE_COL,
    HISTORY_COL,
    SOURCE_COL,
    "request_text",  # Raw text (redundant with edit_aware)
    "requester_username",  # High cardinality ID
    "giver_username_if_known",  # Leakage / Post-hoc
    "post_was_edited",  # Often undefined at prediction time
    "requester_user_flair",  # High missing rate
    "unix_timestamp_of_request_utc",  # Redundant with unix_timestamp_of_request
]

# Suffix for retrieval-time features that must be dropped to prevent leakage
RETRIEVAL_SUFFIX = "_at_retrieval"

# -----------------------------------------------------------------------------
# Model & Training Configuration
# -----------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5

# Pre-trained Sentence Transformer model for dense embeddings
# Using a smaller model (384-dim) to prevent curse of dimensionality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Random Forest Hyperparameters
# Used for: Lexical Bagger, Community Bagger, Semantic Bagger
# Strategy: High estimators, leaf regularization to prevent overfitting on noise
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# XGBoost Hyperparameters
# Used for: Semantic Booster, Persona Booster
# Strategy: Conservative learning rate, depth control, and scale_pos_weight for imbalance
# scale_pos_weight ~ 3.0 based on ~75/25 class distribution
XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}
# Early stopping rounds for XGBoost fit() method
XGB_EARLY_STOPPING_ROUNDS = 50

# Logistic Regression Hyperparameters
# Used for: Metadata Anchor, Meta-Learner (Level 2)
# Strategy: L2 regularization, balanced weights
LR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": SEED,
}

# TF-IDF Vectorizer Configuration
# Used for: Sparse Lexical and Behavioral Views
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "sublinear_tf": True,
    "max_features": 5000,
    "stop_words": "english",
}
