import os
import numpy as np

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_27"
SUBMISSION_DIR = "./submission"

# Create necessary writable directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Data Files
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

# Metadata Files (Generated previously)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Submission File
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths for Pre-computed Embeddings (to avoid re-inference)
# Primary Backbone (MiniLM)
TRAIN_PRIMARY_EMBS_PATH = os.path.join(WORKING_DIR, "train_primary_embeddings.npy")
VAL_PRIMARY_EMBS_PATH = os.path.join(WORKING_DIR, "val_primary_embeddings.npy")
TEST_PRIMARY_EMBS_PATH = os.path.join(WORKING_DIR, "test_primary_embeddings.npy")

# Auxiliary Backbone (MPNet)
TRAIN_AUX_EMBS_PATH = os.path.join(WORKING_DIR, "train_aux_embeddings.npy")
VAL_AUX_EMBS_PATH = os.path.join(WORKING_DIR, "val_aux_embeddings.npy")
TEST_AUX_EMBS_PATH = os.path.join(WORKING_DIR, "test_aux_embeddings.npy")

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================
# Primary Backbone: High Resolution, Compact (384d)
# Used as the "Anchor View"
PRIMARY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Auxiliary Backbone: Low Resolution, Deep Semantics (768d -> Compressed)
# Used for "Asymmetric Dimensionality Reduction"
AUX_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# Text columns to be concatenated and encoded
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Numerical metadata columns for View 3 (Robust Metadata)
# Explicitly including 'unix_timestamp_of_request' as per strategy
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

# Target Label
TARGET_COL = "requester_received_pizza"

# =============================================================================
# HYPERPARAMETERS & TRAINING CONFIGURATION
# =============================================================================
SEED = 42
N_FOLDS = 5

# Dimensionality Reduction Settings
# Compressing Auxiliary View (MPNet) to 32 dimensions
PCA_COMPONENTS = 32

# Bagging Ensemble Settings
N_BAGGING_ESTIMATORS = 20

# Logistic Regression Grid Search Space
# Strategy: High-Regularization Regime (C: 1e-4 to 10.0), L2 Penalty (Ridge)
LR_PARAM_GRID = {
    "C": np.logspace(-4, 1, 6).tolist(),  # [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    "class_weight": ["balanced", None],
    "penalty": ["l2"],
    "solver": ["lbfgs"],  # lbfgs is standard for L2 and generally robust
}

# Runtime / Debugging Control
# Set to an integer (e.g., 100) to run on a small subset for debugging.
# Set to None for full training.
DEBUG_SAMPLE_SIZE = None
