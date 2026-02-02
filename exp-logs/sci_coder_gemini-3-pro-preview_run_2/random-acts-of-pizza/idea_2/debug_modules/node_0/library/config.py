import os

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
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

# Submission Path
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths for Features
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# ==========================================
# Model & Training Configuration
# ==========================================
SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64  # Batch size for generating embeddings on GPU

# Hyperparameter Grid for Logistic Regression (L2 Regularization)
# C is the inverse of regularization strength; smaller values specify stronger regularization.
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# ==========================================
# Feature Configuration
# ==========================================
# Text columns to be concatenated and encoded
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Numerical features to be scaled and used in the tabular branch
NUMERICAL_FEATURES = [
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
