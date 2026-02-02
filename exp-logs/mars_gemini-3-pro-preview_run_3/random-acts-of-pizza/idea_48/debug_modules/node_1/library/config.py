import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_48"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Specific File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"
N_FOLDS = 5

# =============================================================================
# FEATURE SELECTION
# =============================================================================

# Text Columns for Concatenation (Title + Edit-Aware Body)
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Metadata Allow-List
# Strictly excludes in-domain history (e.g., posts_on_raop) and retrieval-time leakage.
# Includes raw timestamp for temporal profiling.
ALLOW_LIST_META = [
    "unix_timestamp_of_request_utc",
    "requester_account_age_in_days_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
]

# =============================================================================
# VECTORIZATION & EMBEDDING CONFIG
# =============================================================================

# TF-IDF Configuration for Lexical Branch
TFIDF_PARAMS = {
    "sublinear_tf": True,
    "min_df": 5,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "max_features": 20000,  # Cap features to prevent memory explosion
}

# Count Vectorizer Config for Community Branch (Subreddit History)
COMMUNITY_VEC_PARAMS = {
    "max_features": 1000,  # Top 1000 subreddits only (Sparse representation)
    "binary": True,  # Bag-of-Concepts approach
    "token_pattern": r"(?u)\b\w+\b",  # Simple tokenization for subreddit names
}

# Embedding Model Name (HuggingFace)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Lexical Bagger (Random Forest)
# High regularization (min_samples_leaf=2) to prevent overfitting on sparse text data.
RF_PARAMS_LEXICAL = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 2. Community Bagger (Random Forest)
# Modeled on sparse subreddit history.
RF_PARAMS_COMMUNITY = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 3. Semantic Booster (XGBoost)
# Captures non-linear signals in dense embedding space.
# scale_pos_weight ~3.02 derived from training class imbalance (75/25).
XGB_PARAMS_SEMANTIC = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.02,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "eval_metric": "auc",
}

# 4. Semantic Bagger (Random Forest)
# Structural diversity for dense embeddings.
# Constrained depth to prevent memorization of embedding noise.
RF_PARAMS_SEMANTIC = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 5. Metadata Anchor (Logistic Regression)
# High-bias regularizer for tabular metadata.
LR_PARAMS_ANCHOR = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": SEED,
}

# 6. Temporal Booster (LightGBM)
# Non-linear metadata model to capture temporal drift via raw timestamp splits.
LGBM_PARAMS_TEMPORAL = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    "metric": "auc",
    "early_stopping_rounds": 50,
}

# 7. Level 2 Meta-Learner (Logistic Regression)
# Stacking calibrator.
META_LEARNER_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "random_state": SEED,
}
