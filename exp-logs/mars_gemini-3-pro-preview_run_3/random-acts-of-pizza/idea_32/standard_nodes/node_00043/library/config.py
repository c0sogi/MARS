import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_32")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
N_FOLDS = 5
VAL_SIZE = 0.2
EARLY_STOPPING_ROUNDS = 50

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Text Processing
TEXT_VOCAB_SIZE = 3000
HISTORY_VOCAB_SIZE = 1000
TEXT_MIN_DF = 5
NGRAM_RANGE = (1, 2)

# Dense Embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Global Metadata Vector - Allow List
# Includes raw timestamp for temporal drift, excludes retrieval-time and derived text stats.
METADATA_FEATURES = [
    "unix_timestamp_of_request_utc",  # Critical Temporal Anchor
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
]

# Target Column
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Sparse Lexical Branch (Text TFIDF + Metadata) -> Random Forest
# Regularization: min_samples_leaf=2 to prevent overfitting on sparse features
LEXICAL_RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 2. Sparse Behavioral Branch (History TFIDF + Metadata) -> Random Forest
# Regularization: min_samples_leaf=2, restricted vocabulary (1000)
BEHAVIORAL_RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 3. Dense Semantic Branch (Embeddings + Metadata) -> XGBoost
# High capacity with early stopping
SEMANTIC_XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 4,  # Limit threads to avoid contention
    "random_state": SEED,
    "verbosity": 0,
    # scale_pos_weight is usually calculated dynamically: neg_count / pos_count (~3.0)
    "scale_pos_weight": 3.0,
}

# 4. Dense Semantic Branch (Embeddings + Metadata) -> Random Forest
# Topology-Specific Regularization: Stricter constraints for dense continuous data
SEMANTIC_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,  # Constrained to prevent memorizing noise in embedding space
    "min_samples_leaf": 4,  # Higher threshold than sparse branches
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": 0,
}

# 5. Contextual Branch (Metadata Only) -> Logistic Regression
# High-bias regularizer
CONTEXTUAL_LR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": SEED,
    "max_iter": 1000,
}

# Level 2 Meta-Learner -> Logistic Regression
META_LEARNER_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "random_state": SEED,
    "fit_intercept": True,
}
