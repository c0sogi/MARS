import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_15")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

RANDOM_SEED = 42
N_FOLDS = 5
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Column Definitions
TEXT_COL = "request_text_edit_aware"  # Use edit-aware text to prevent leakage
TITLE_COL = "request_title"
SUBREDDIT_COL = "requester_subreddits_at_request"

# Numerical Features (Global Metadata Vector)
# These are raw columns available at inference time (request time)
RAW_NUMERICAL_COLS = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "unix_timestamp_of_request_utc",
]

# Lexical View (Sparse Text)
LEXICAL_PARAMS = {
    "max_features": 3000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "sublinear_tf": True,
    "stop_words": "english",
}

# Behavioral View (Sparse History)
BEHAVIORAL_PARAMS = {
    "max_features": 1000,
    "ngram_range": (1, 1),
    "min_df": 2,
    "sublinear_tf": True,
    "stop_words": None,  # Subreddit names are the tokens
}

# Semantic View (Dense Embeddings)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Lexical Bagger (Random Forest on Text + Metadata)
L1_RF_LEXICAL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,  # Regularization
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 2. Behavioral Bagger (Random Forest on Subreddits + Metadata)
L1_RF_BEHAVIORAL_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 3. Semantic Booster (XGBoost on Embeddings + Metadata)
L1_XGB_SEMANTIC_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": 3.0,  # Handle imbalance (~1:3)
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}
XGB_EARLY_STOPPING_ROUNDS = 50

# 4. Semantic Bagger (Random Forest on Embeddings + Metadata)
L1_RF_SEMANTIC_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,  # Restrict depth on dense features
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 5. Contextual Baseline (Logistic Regression on Metadata only)
L1_LOGREG_CONTEXTUAL_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 2000,
    "random_state": RANDOM_SEED,
    "verbose": 0,
}

# Level 2 Meta-Learner (Logistic Regression Stacker)
L2_META_PARAMS = {
    "C": 0.1,  # Strong regularization
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": None,  # Rely on calibrated probabilities
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "verbose": 0,
}
