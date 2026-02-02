import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Input Data (Metadata Parquet files)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

# Output Submission
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Caching Paths (for deterministic processing)
CACHE_TRAIN_PROCESSED = os.path.join(WORKING_DIR, "train_processed.parquet")
CACHE_VAL_PROCESSED = os.path.join(WORKING_DIR, "val_processed.parquet")
CACHE_TEST_PROCESSED = os.path.join(WORKING_DIR, "test_processed.parquet")

# Feature Cache (numpy/npz for arrays)
CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "X_train_features.npz")
CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "X_val_features.npz")
CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "X_test_features.npz")

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
SEED = 42
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"
TEXT_COL = "request_text_edit_aware"
TITLE_COL = "request_title"
SUBREDDIT_COL = "requester_subreddits_at_request"

# Columns to exclude to prevent leakage or redundancy
# Note: Columns ending in "_at_retrieval" are handled dynamically in processing
DROP_COLS = [
    "request_id",
    "requester_received_pizza",
    "request_text",  # Using edit_aware version
    "request_text_edit_aware",  # Processed into features
    "request_title",  # Processed into features
    "requester_subreddits_at_request",  # Processed into features
    "requester_username",
    "giver_username_if_known",  # Target leakage
    "source_file",
    "unix_timestamp_of_request",
    "unix_timestamp_of_request_utc",
    "post_was_edited",  # Often updated post-request
    "requester_user_flair",  # Updated upon success
]

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================
# Lexical View (TF-IDF on Request Text)
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.9

# Semantic View (SBERT Embeddings)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 32
SBERT_DIM = 384

# Community View (Subreddit History -> TF-IDF -> SVD)
SUBREDDIT_TFIDF_MAX_FEATURES = 1000
SVD_COMPONENTS = 20
SVD_RANDOM_STATE = SEED

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# General
N_JOBS = 12  # 12 vCPUs available
N_FOLDS = 5  # For Stacking CV

# Level 1: Lexical Bagger (Random Forest)
# Optimized for sparse, high-dimensional text data
L1_LEXICAL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": 0,
}

# Level 1: Semantic Bagger (Random Forest)
# Optimized for dense embeddings (depth limited to prevent overfitting)
L1_SEMANTIC_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": 0,
}

# Level 1: Community Booster (XGBoost)
# Optimized for dense, structured user profile data
L1_COMMUNITY_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbosity": 0,
    # scale_pos_weight is calculated dynamically during training
}

# Level 2: Meta-Learner (Logistic Regression)
# Calibrates the ensemble probabilities
L2_META_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": None,
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

# =============================================================================
# DEBUGGING & RUNTIME
# =============================================================================
DEBUG = False
DEBUG_SAMPLE_SIZE = 200
