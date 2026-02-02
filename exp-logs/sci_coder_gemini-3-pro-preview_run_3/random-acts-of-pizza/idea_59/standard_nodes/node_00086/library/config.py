import os
import numpy as np

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Random Seed for Reproducibility
SEED = 42
np.random.seed(SEED)

# Compute Resources
N_JOBS = 12  # Available vCPUs

# =============================================================================
# FILE PATHS
# =============================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_59"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths (Parquet Metadata)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA COLUMN DEFINITIONS
# =============================================================================

ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text Columns for Concatenation
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Community/Behavioral Column
COMMUNITY_COL = "requester_subreddits_at_request"

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Granular Tokenization Pattern (Captures "I", "$", etc.)
TOKEN_PATTERN = r"\w{1,}"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Metadata Feature Allow-List (Hygienic Feature Selection)
# Explicitly including restored RAOP history and raw timestamps
ALLOW_LIST = [
    # Temporal
    "unix_timestamp_of_request_utc",
    # Restored RAOP History (Pre-request signals)
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    # User Statistics (Agency/Status)
    "requester_account_age_in_days_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
]

# Columns to explicitly exclude (Leakage Prevention)
EXCLUDE_SUFFIX = "_at_retrieval"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Sparse Lexical Branch (Granular Lexical Bagger)
# Random Forest on TF-IDF (Title + Body)
LEXICAL_BAGGER_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbose": 0,
}

# TF-IDF Vectorizer Params for Lexical Branch
LEXICAL_VECTORIZER_PARAMS = {
    "strip_accents": "unicode",
    "stop_words": "english",
    "token_pattern": TOKEN_PATTERN,  # Granular
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 5,
    "max_features": 10000,
}

# 2. Sparse Behavioral Branch (Community Bagger)
# Random Forest on TF-IDF (Subreddits)
COMMUNITY_BAGGER_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

# TF-IDF Vectorizer Params for Community Branch
COMMUNITY_VECTORIZER_PARAMS = {
    "strip_accents": "unicode",
    "token_pattern": TOKEN_PATTERN,
    "binary": True,  # Set of concepts
    "max_features": 1000,  # Strict vocabulary limit
}

# 3. Dense Semantic Branch (Text Embeddings)

# 3a. Semantic Booster (XGBoost) - Conservative
SEMANTIC_BOOSTER_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,  # Conservative
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.6,  # Regularization
    "scale_pos_weight": 3.0,  # Imbalance handling (~75/25)
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "tree_method": "hist",
    "early_stopping_rounds": 100,
}

# 3b. Semantic Gradient (LightGBM) - Leaf-wise
SEMANTIC_GRADIENT_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbose": -1,
}

# 3c. Semantic Bagger (Random Forest) - Structural Diversity
SEMANTIC_BAGGER_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,  # Modality-specific regularization
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

# 4. Contextual Branch (Metadata)

# 4a. Metadata Anchor (Logistic Regression) - High Bias
METADATA_ANCHOR_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": SEED,
}

# 4b. Temporal Booster (LightGBM) - Non-linear Temporal Drift
TEMPORAL_BOOSTER_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.02,
    "max_depth": 3,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbose": -1,
}

# 5. Level 2 Meta-Learner
META_LEARNER_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "random_state": SEED,
}

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

N_FOLDS = 5
