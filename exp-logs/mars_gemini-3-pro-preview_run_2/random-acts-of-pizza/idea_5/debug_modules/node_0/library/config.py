import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Paths
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
FINE_TUNED_MODEL_PATH = os.path.join(WORKING_DIR, "fine_tuned_transformer")

# Cache Paths (for intermediate steps)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# ==========================================
# Data Configuration
# ==========================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text columns to be concatenated for embedding
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Numerical columns (Strictly 'at_request' to avoid look-ahead bias)
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

# ==========================================
# Model Hyperparameters
# ==========================================
SEED = 42

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Stage 1: Siamese Transformer Fine-Tuning ---
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 384  # Max length for the tokenizer
FINE_TUNE_BATCH_SIZE = 16  # Small batch size for effective triplet mining
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LR = 2e-5
TRIPLET_MARGIN = 0.5

# --- Stage 2: Linear Classifier (Logistic Regression) ---
# Grid search for regularization strength C.
# Includes strong regularization (small C) as per strategy to prevent overfitting.
CLASSIFIER_C_GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
CLASSIFIER_SOLVER = "liblinear"  # Good for smaller datasets
CLASSIFIER_MAX_ITER = 1000
CV_FOLDS = 5
