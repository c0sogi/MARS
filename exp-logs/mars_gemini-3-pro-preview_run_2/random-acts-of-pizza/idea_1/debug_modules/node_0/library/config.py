import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Raw Data
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

# Metadata (Splits)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Output
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Files for Processed Features (Parquet format)
TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG_MODE is True

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# Text Features
# We combine title and text, then apply Hashing Trick
TEXT_COLS = ["request_title", "request_text_edit_aware"]
HASH_VECTOR_SIZE = 128  # Dimension size for HashingVectorizer

# Numerical Features
# Selected features that are available at the time of request (preventing leakage)
NUMERIC_COLS = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
]

# Target Variable
TARGET_COL = "requester_received_pizza"

# =============================================================================
# MODEL HYPERPARAMETERS (Random Forest)
# =============================================================================
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": 12,  # Use available vCPUs
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",  # Handle class imbalance (approx 1:3)
    "verbose": 0,
}
