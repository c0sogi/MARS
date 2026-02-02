import os

# =============================================================================
# GLOBAL CONFIGURATION & PATHS
# =============================================================================

SEED = 42
NUM_FOLDS = 5

# Directory Layout
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_52"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# =============================================================================
# DATA COLUMNS & FEATURES
# =============================================================================

ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text Modality: Concatenation of Title and Edit-Aware Body
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Behavioral Modality: Subreddit History
SUBREDDIT_COL = "requester_subreddits_at_request"

# Contextual Modality: Allow-listed Numerical Features (Restored Domain Priors)
# Explicitly including RAOP history and raw timestamps, excluding derived text lengths
NUMERICAL_COLS = [
    "unix_timestamp_of_request_utc",
    "requester_account_age_in_days_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
    # Restored RAOP History (Reciprocity Signals)
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_days_since_first_post_on_raop_at_request",
]

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Lexical Branch: Sparse TF-IDF
TFIDF_PARAMS = {
    "sublinear_tf": True,
    "min_df": 5,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "max_features": 10000,
}

# Behavioral Branch: Bag-of-Concepts
COMMUNITY_PARAMS = {
    "binary": True,
    "max_features": 1000,
    "token_pattern": r"(?u)\b\w+\b",
}

# Semantic Branch: Dense Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================================
# MODEL HYPERPARAMETERS (LEVEL 1: BASE LEARNERS)
# =============================================================================

# 1. Sparse Lexical Branch (Text Modality)
# High regularization to prevent overfitting on sparse features
LEXICAL_BAGGER_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "max_depth": None,
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 2. Sparse Behavioral Branch (History Modality)
# Limited vocabulary to prevent overfitting to rare communities
COMMUNITY_BAGGER_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "max_depth": None,
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 3. Dense Semantic Branch (Text Modality) - Volatile Learner
# XGBoost on embeddings
SEMANTIC_BOOSTER_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,  # Handling class imbalance
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,  # For training loop
    "verbosity": 0,
}

# 4. Dense Semantic Branch (Text Modality) - Volatile Learner
# LightGBM on embeddings for algorithmic diversity
SEMANTIC_GRADIENT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    "class_weight": "balanced",
    "early_stopping_rounds": 50,
}

# 5. Dense Semantic Branch (Text Modality) - Stable Learner
# Random Forest on embeddings with depth constraints
SEMANTIC_BAGGER_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 6. Contextual Branch (Metadata Modality) - Stable Learner
# Linear Anchor
METADATA_ANCHOR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 7. Contextual Branch (Metadata Modality) - Volatile Learner
# Non-linear temporal booster
TEMPORAL_BOOSTER_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "random_state": SEED,
    "verbose": -1,
    "class_weight": "balanced",
    "early_stopping_rounds": 50,
}

# =============================================================================
# MODEL HYPERPARAMETERS (LEVEL 2: META-LEARNER)
# =============================================================================

META_LEARNER_PARAMS = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": SEED,
}
