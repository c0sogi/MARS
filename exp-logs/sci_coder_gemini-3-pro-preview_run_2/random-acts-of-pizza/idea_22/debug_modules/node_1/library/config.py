import os

# ==========================================
# Global Configuration & Reproducibility
# ==========================================
SEED = 42
N_FOLDS = 5
N_JOBS = 12  # Use available vCPUs

# Debugging settings
DEBUG = False
DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True

# ==========================================
# File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Cache Directory for deterministic processing (e.g., embeddings)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_22")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Files
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Submission
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Feature Configuration
# ==========================================
# SBERT Model for Text Embeddings
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text Columns to Concatenate (Title + Edit-Aware Text)
TEXT_COLS_TO_CONCAT = ["request_title", "request_text_edit_aware"]

# Numerical Metadata Features
# Strategy requires Early Fusion of these with embeddings.
# Explicitly including 'unix_timestamp_of_request' as per strategy.
# Excluding 'unix_timestamp_of_request_utc' due to perfect collinearity.
# Excluding explicit user history embeddings/text, keeping only count metadata.
NUMERIC_FEATURES = [
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
# Search spaces focus on the "High-Regularization Regime" to prevent overfitting.

# 1. Logistic Regression (Log-Likelihood Optimization)
# C: Inverse of regularization strength (smaller = stronger regularization).
LR_GRID = {
    "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
    "class_weight": ["balanced", None],
    "penalty": ["l2"],
    "solver": ["liblinear"],
}

# 2. Linear SVM (Hinge Loss Optimization)
# Implemented via SGDClassifier.
# alpha: Regularization strength (larger = stronger regularization).
# Note: C approx 1/alpha.
SVM_GRID = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "penalty": ["l2", "l1", "elasticnet"],
    "class_weight": ["balanced", None],
    "loss": ["hinge"],
    "max_iter": [2000],
}

# 3. Ridge Classifier (Squared Error Optimization)
# alpha: Regularization strength (larger = stronger regularization).
RIDGE_GRID = {
    "alpha": [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0],
    "class_weight": ["balanced", None],
}

# Bagging Configuration for Base Learners
BAGGING_CONFIG = {
    "n_estimators": 10,
    "max_samples": 1.0,
    "bootstrap": True,
    "random_state": SEED,
    "n_jobs": 1,  # Avoid nested parallelism conflicts
}
