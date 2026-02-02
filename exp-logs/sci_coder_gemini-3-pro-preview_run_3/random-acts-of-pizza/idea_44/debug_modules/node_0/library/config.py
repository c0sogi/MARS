import os

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SEED = 42
N_FOLDS = 5

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_44"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"

# Input Data Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA COLUMNS
# =============================================================================

TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# Text Columns for Concatenation
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Subreddit History Column
SUBREDDIT_COL = "requester_subreddits_at_request"

# Explicit Allow-List for Metadata (Dense Features)
# Includes Raw Timestamp and User Stats at Request Time.
# Excludes all '_at_retrieval' columns to prevent leakage.
METADATA_COLS = [
    "unix_timestamp_of_request_utc",
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
]

# =============================================================================
# FEATURE EXTRACTION CONFIG
# =============================================================================

# Lexical Features (TF-IDF on Concatenated Text)
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "sublinear_tf": True,
    "max_features": 5000,
    "stop_words": "english",
}

# Community Features (TF-IDF on Subreddit List)
SUBREDDIT_TFIDF_PARAMS = {
    "max_features": 1000,
    "binary": True,
    "stop_words": None,  # Subreddit names are the tokens
}

# Semantic Features (Dense Embeddings)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Lexical Bagger (Random Forest on TF-IDF + Metadata)
RF_LEXICAL_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 2. Community Bagger (Random Forest on Subreddit History + Metadata)
RF_COMMUNITY_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 3. Semantic Booster (XGBoost on Embeddings + Metadata)
XGB_SEMANTIC_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,  # Approximate imbalance ratio
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

# 4. Semantic Bagger (Random Forest on Embeddings + Metadata)
RF_SEMANTIC_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 5. Metadata Anchor (Logistic Regression on Metadata)
LR_METADATA_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": SEED,
}

# 6. Temporal Booster (LightGBM on Metadata - Non-Linear)
LGBM_TEMPORAL_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
}

# Level 2 Meta-Learner (Logistic Regression)
META_LEARNER_PARAMS = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": SEED,
}
