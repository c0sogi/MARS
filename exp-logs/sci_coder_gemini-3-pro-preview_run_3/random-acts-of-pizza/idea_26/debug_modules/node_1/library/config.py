import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_26"
SUBMISSION_DIR = "./submission"

# Sub-directories for artifacts
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

SEED = 42
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"
TEXT_COL = "request_text_edit_aware"
TITLE_COL = "request_title"
SUBREDDIT_LIST_COL = "requester_subreddits_at_request"

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Transformer model for dense embeddings (Compact model)
TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# TF-IDF Parameters for Sparse Views
TFIDF_PARAMS = {
    "max_features": 3000,
    "sublinear_tf": True,
    "min_df": 5,
    "stop_words": "english",
    "ngram_range": (1, 2),
}

# Allow-listed Metadata Features (Dense Numerical)
# Explicitly excluding derived length features and retrieval-time leakage
METADATA_FEATURES = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "unix_timestamp_of_request",  # Temporal feature
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Sparse Lexical Branch (Text Modality) - Random Forest
LEXICAL_BAGGER_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 2. Sparse Behavioral Branch (History Modality) - Random Forest
COMMUNITY_BAGGER_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 3. Dense Semantic Branch (Text Modality) - XGBoost
# Note: scale_pos_weight is often calculated dynamically, but can be set here if fixed.
# Early stopping is handled in the training loop logic using these params.
SEMANTIC_BOOSTER_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": SEED,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    # 'scale_pos_weight': 3.0, # Approximate ratio, can be dynamic in code
}
SEMANTIC_BOOSTER_FIT_PARAMS = {"early_stopping_rounds": 50, "verbose": False}

# 4. Dense Semantic Branch (Text Modality) - Random Forest (Bagging on Dense)
SEMANTIC_BAGGER_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "max_features": "sqrt",  # Standard for dense inputs
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 5. Contextual Branch (Metadata Modality) - Logistic Regression
METADATA_ANCHOR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": SEED,
}

# Level 2: Meta-Learner - Logistic Regression
META_LEARNER_PARAMS = {
    "C": 0.1,  # Stronger regularization for stacking
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": SEED,
}
