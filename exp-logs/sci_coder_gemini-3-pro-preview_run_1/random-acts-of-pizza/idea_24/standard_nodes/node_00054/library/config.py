import os
import torch

# -----------------------------------------------------------------------------
# Global Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_24"
SUBMISSION_DIR = "./submission"

# Ensure essential directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Column Definitions & Feature Lists
# -----------------------------------------------------------------------------
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text Columns
# We use the edit-aware text to avoid leakage (e.g. "EDIT: Thanks for the pizza")
TEXT_TITLE_COL = "request_title"
TEXT_BODY_COL = "request_text_edit_aware"
TEXT_COLS = [TEXT_TITLE_COL, TEXT_BODY_COL]

# History/List Column (List of subreddits)
HISTORY_COL = "requester_subreddits_at_request"

# Numerical Features
# Strictly restricted to the intersection of Train and Test schemas to prevent leakage.
# Excludes '_at_retrieval' columns which are not present in the test set.
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

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Feature Engineering Parameters
# -----------------------------------------------------------------------------
# TF-IDF Configuration (Stream A)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Bayesian Target Encoding (Stream A)
# Smoothing parameter (k): balances observed mean with global prior.
# Higher k = stronger regularization towards global mean for rare subreddits.
BAYESIAN_SMOOTHING_K = 10

# SBERT Configuration (Stream B)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 32
SBERT_EMBEDDING_DIM = 384

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Stream A: Random Forest
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Stream B: Masked-Attention Gated MLP
MAX_HISTORY_LEN = 50  # Fixed sequence length for history padding

MLP_PARAMS = {
    "hidden_dim": 128,
    "dropout": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "patience": 15,
    "embedding_dim": SBERT_EMBEDDING_DIM,
}
