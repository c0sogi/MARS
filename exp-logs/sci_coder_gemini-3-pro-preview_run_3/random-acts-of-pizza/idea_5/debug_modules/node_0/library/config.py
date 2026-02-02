import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Create working and submission directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
N_FOLDS = 5
DEBUG = False  # Set to True for fast debugging runs
DEBUG_SIZE = 100  # Number of samples to use in debug mode

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text Columns
# We use the edit-aware text to avoid leakage from edits like "EDIT: Thanks for pizza!"
TEXT_COL = "request_text_edit_aware"
TITLE_COL = "request_title"

# Behavioral Columns
SUBREDDIT_COL = "requester_subreddits_at_request"

# Numerical Columns (Strictly available at request time to prevent leakage)
NUMERICAL_COLS = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "unix_timestamp_of_request",
]

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)

SVD_COMPONENTS = 20  # For reducing subreddit sparse matrix to dense user persona

SBERT_MODEL = "all-MiniLM-L6-v2"  # Efficient, high-performance sentence transformer

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Lexical Bagger (Random Forest on TF-IDF + Meta)
# Random Forest is robust for high-dimensional sparse data
LEXICAL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# 2. Semantic Bagger (Random Forest on SBERT Embeddings + Meta)
# Random Forest handles the non-linearities of dense embeddings well without fine-tuning
SEMANTIC_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# 3. Community Booster (XGBoost on Subreddit SVD + Meta)
# XGBoost excels at dense, structured features like SVD components
COMMUNITY_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,  # Handle class imbalance (approx 0.75/0.25)
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
}

# 4. Meta Learner (Logistic Regression Stacking)
# Linear meta-learner to calibrate and combine probabilities
META_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": SEED,
    "max_iter": 1000,
}
