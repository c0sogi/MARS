import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for the specific idea (Idea 19)
WORKING_DIR = "./working/idea_19"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
NUM_FOLDS = 5
# Approximate scale_pos_weight based on 25% positive class (75/25 = 3)
SCALE_POS_WEIGHT = 3.0

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text Columns
# We use the edit-aware text to prevent leakage from "EDIT: Thanks" messages
TEXT_COL = "request_text_edit_aware"
TITLE_COL = "request_title"

# Behavioral/Community Columns
SUBREDDIT_LIST_COL = "requester_subreddits_at_request"

# Numerical/Metadata Allow-list
# Strictly using features available AT REQUEST time.
# Explicitly excluding explicit text length features as per Lesson 00027.
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
    # Timestamp used for temporal feature extraction (Hour, Day)
    "unix_timestamp_of_request_utc",
]

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================

# TF-IDF for Request Text
TEXT_TFIDF_PARAMS = {
    "max_features": 3000,
    "min_df": 5,
    "sublinear_tf": True,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "strip_accents": "unicode",
}

# TF-IDF for Subreddit History (Bag-of-Concepts)
SUBREDDIT_TFIDF_PARAMS = {
    "max_features": 1000,
    "min_df": 5,
    "sublinear_tf": True,
    "ngram_range": (1, 1),
    "stop_words": None,
    "strip_accents": "unicode",
}

# Dense Embedding Model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Level 1: Sparse Bagging (Random Forest)
# Used for Lexical Bagger and Community Bagger
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,  # Regularization per Lesson 00025
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# Level 1: Dense Hybrid (XGBoost)
# Used for Semantic Hybrid and Persona Hybrid
XGB_PARAMS = {
    "n_estimators": 2000,  # High number, controlled by early stopping
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
    # Early stopping rounds to be passed to fit(), not init
    "early_stopping_rounds": 50,
}

# Level 1: Dense Hybrid (Random Forest component)
# Used alongside XGBoost in the dense branches
RF_DENSE_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# Level 1: Metadata Anchor (Logistic Regression)
LR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
}

# Level 2: Meta-Learner (Stacking)
META_LEARNER_PARAMS = {
    "C": 0.1,  # Stronger regularization for the meta-learner
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": None,  # Let the probabilities speak for themselves
    "random_state": RANDOM_SEED,
    "fit_intercept": True,
}
