import os

# ==========================================
# Path Configuration
# ==========================================
METADATA_DIR = "./metadata"
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Working directory for caching intermediate files
WORKING_DIR = "./working/idea_1"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ==========================================
# Global Constants
# ==========================================
RANDOM_STATE = 42
DEBUG_SAMPLE_SIZE = (
    None  # Set to an integer (e.g., 100) for debugging with smaller data
)

# ==========================================
# Data Definitions
# ==========================================
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# Text features to be combined and vectorized
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Numerical features available in both training and inference (test) time.
# These strictly exclude 'at_retrieval' features to prevent data leakage.
NUMERIC_COLS = [
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

# ==========================================
# Model Hyperparameters
# ==========================================
# Random Forest Classifier parameters
RF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,  # Allow full depth as per baseline suggestion, or tune later
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",  # Handles the ~25% positive class imbalance
    "random_state": RANDOM_STATE,
    "n_jobs": -1,  # Use all available vCPUs
    "verbose": 0,
}

# XGBoost Classifier parameters
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
    "eval_metric": "auc",
}

# CountVectorizer parameters for Bag-of-Words
VECTORIZER_PARAMS = {
    "max_features": 2000,
    "stop_words": "english",
    "binary": False,  # False = count occurrences, True = binary presence
    "strip_accents": "unicode",
}
