import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_37")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_STATE = 42
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# Text columns to be concatenated for processing
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Explicit Allow-List for Metadata (Numerical/Dense Features)
# Includes Raw Temporal Anchors and User Stats
METADATA_COLS = [
    "unix_timestamp_of_request_utc",
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

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# 1. Sparse Text Branch (TF-IDF)
TEXT_TFIDF_PARAMS = {
    "sublinear_tf": True,
    "min_df": 5,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "max_features": 20000,  # Cap features to keep memory reasonable
}

# 2. Sparse Behavioral Branch (Community Bag-of-Concepts)
# Constrained to Top 1,000 subreddits
COMMUNITY_TFIDF_PARAMS = {
    "max_features": 1000,
    "binary": True,  # Presence/Absence is more robust for history
    "stop_words": None,
}

# 3. Dense Text Branch (Embeddings)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
N_FOLDS = 5

# Base Learner 1: Enhanced Lexical Bagger (Sparse Text + Meta -> RF)
MODEL_LEXICAL_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,  # Regularization for sparse features
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Base Learner 2: Constrained Community Bagger (Sparse Behavioral + Meta -> RF)
MODEL_COMMUNITY_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Base Learner 3: Semantic Booster (Dense Text + Meta -> XGBoost)
# Note: scale_pos_weight will be calculated dynamically in the pipeline
MODEL_SEMANTIC_XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 4,  # Limit threads per model
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 50,
    "eval_metric": "auc",
}

# Base Learner 4: Semantic Bagger (Dense Text + Meta -> RF)
# Modality-Specific Regularization
MODEL_SEMANTIC_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,  # Prevent memorization of embedding noise
    "min_samples_leaf": 4,  # Stronger leaf regularization
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Base Learner 5: Contextual Anchor (Meta -> Logistic Regression)
MODEL_META_ANCHOR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Level 2: Stacking Meta-Learner (Logistic Regression)
STACKING_META_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}
