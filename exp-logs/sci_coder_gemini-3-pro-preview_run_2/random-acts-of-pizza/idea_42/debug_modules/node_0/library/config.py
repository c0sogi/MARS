import os

# =============================================================================
# Global Configuration
# =============================================================================

# Random Seed for Reproducibility
SEED = 42

# =============================================================================
# Directory Paths
# =============================================================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
# Using idea_42 as the working directory for this specific strategy
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_42")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# File Paths
# =============================================================================
# Raw Data
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

# Metadata (Generated previously)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Output
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Feature Configuration
# =============================================================================
# Target Variable
TARGET_COL = "requester_received_pizza"

# Text Columns for Multi-View Encoding
TEXT_COL_TITLE = "request_title"
TEXT_COL_BODY = "request_text_edit_aware"

# Numerical Metadata Features
# Selected based on robustness and availability at inference time
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
    "unix_timestamp_of_request",  # Explicitly included for temporal signal
]

# =============================================================================
# Model & Pipeline Hyperparameters
# =============================================================================

# Debugging / Development
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG_MODE is True

# Embedding Models (HuggingFace)
# High-resolution anchors
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Deep context auxiliary
MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Dimensionality Reduction (Whitened PCA)
# Used for the Global Context view to normalize variance of semantic signals
PCA_COMPONENTS = 50
PCA_WHITEN = True

# Cross-Validation Strategy
N_FOLDS = 5

# Ensemble Classifier (Bagged Logistic Regression)
N_ESTIMATORS = 20  # Number of base estimators in the Bagging ensemble

# Hyperparameter Search Space (for Grid Search on the Base Estimator)
# Note: These values are applied to the LogisticRegression base estimator
GRID_C = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
GRID_CLASS_WEIGHT = ["balanced", None]
GRID_PENALTY = ["l2"]
GRID_SOLVER = ["lbfgs"]

# =============================================================================
# Caching Configuration
# =============================================================================
# Paths for cached intermediate files to speed up iterative development
CACHE_FILES = {
    # Train Split
    "train_title_emb": os.path.join(WORKING_DIR, "train_title_minilm.npy"),
    "train_body_emb": os.path.join(WORKING_DIR, "train_body_minilm.npy"),
    "train_global_emb": os.path.join(WORKING_DIR, "train_global_mpnet.npy"),
    "train_meta_features": os.path.join(WORKING_DIR, "train_meta.npy"),
    # Validation Split
    "val_title_emb": os.path.join(WORKING_DIR, "val_title_minilm.npy"),
    "val_body_emb": os.path.join(WORKING_DIR, "val_body_minilm.npy"),
    "val_global_emb": os.path.join(WORKING_DIR, "val_global_mpnet.npy"),
    "val_meta_features": os.path.join(WORKING_DIR, "val_meta.npy"),
    # Test Split
    "test_title_emb": os.path.join(WORKING_DIR, "test_title_minilm.npy"),
    "test_body_emb": os.path.join(WORKING_DIR, "test_body_minilm.npy"),
    "test_global_emb": os.path.join(WORKING_DIR, "test_global_mpnet.npy"),
    "test_meta_features": os.path.join(WORKING_DIR, "test_meta.npy"),
}
