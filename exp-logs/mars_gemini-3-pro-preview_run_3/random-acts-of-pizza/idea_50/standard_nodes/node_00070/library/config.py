import os

# -----------------------------------------------------------------------------
# Global Path & Environment Configuration
# -----------------------------------------------------------------------------

SEED = 42
N_FOLDS = 5

# Directory Paths
# Note: Input is read-only. Metadata is pre-generated.
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_50"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Feature Selection & Engineering Configuration
# -----------------------------------------------------------------------------

TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# Text Inputs
# We concatenate Title and Edit-Aware Body for maximum signal
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Community Input
SUBREDDIT_COL = "requester_subreddits_at_request"

# Metadata Allow-List
# Strictly selected to prevent leakage (removing _at_retrieval) and restore valid priors.
# Includes Raw Timestamp for temporal modeling and RAOP history for reciprocity signals.
ALLOW_LIST_METADATA = [
    "unix_timestamp_of_request_utc",
    "requester_account_age_in_days_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
    # Restored RAOP History (Valid Priors)
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_days_since_first_post_on_raop_at_request",
]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------

# 1. Lexical Bagger (Random Forest)
# Input: TF-IDF (Title + Body) + Metadata
# Strategy: High capacity but regularized leaf nodes
LEXICAL_RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 2. Community Bagger (Random Forest)
# Input: TF-IDF (Subreddits) + Metadata
# Strategy: Sparse representation of user history
COMMUNITY_RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 3. Semantic Booster (XGBoost)
# Input: Dense Embeddings + Metadata
# Strategy: Gradient boosting on continuous semantic space
SEMANTIC_XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,  # Handle class imbalance
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
}

# 4. Semantic Bagger (Random Forest)
# Input: Dense Embeddings + Metadata
# Strategy: Structural diversity via bagging, depth-limited to prevent dense noise memorization
SEMANTIC_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,  # Modality-specific regularization
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 5. Metadata Anchor (Logistic Regression)
# Input: Metadata only
# Strategy: High-bias linear baseline
METADATA_ANCHOR_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": SEED,
}

# 6. Temporal Booster (LightGBM)
# Input: Metadata only
# Strategy: Tree-based model to capture non-linear timestamp interactions and drift
TEMPORAL_LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": -1,
    "scale_pos_weight": 3.0,
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
}

# Level 2 Meta-Learner (Logistic Regression)
# Input: Predictions from Level 1 models
META_LEARNER_PARAMS = {
    "penalty": "l2",
    "C": 0.1,  # Strong regularization for stacking
    "class_weight": None,  # Stacking inputs are probabilities
    "solver": "liblinear",
    "random_state": SEED,
}

# -----------------------------------------------------------------------------
# Vectorization & Embedding Configuration
# -----------------------------------------------------------------------------

# TF-IDF for Text (Lexical Branch)
LEXICAL_TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_features": 10000,
    "sublinear_tf": True,
    "stop_words": "english",
}

# TF-IDF for Community (Behavioral Branch)
# Treat subreddit history as a Bag-of-Concepts
COMMUNITY_TFIDF_PARAMS = {
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_features": 1000,  # Strict vocabulary limit
    "binary": True,
    "stop_words": None,
    "tokenizer": lambda x: x,  # Input is already a list of strings
    "preprocessor": lambda x: x,
    "token_pattern": None,
}

# Dense Embeddings (Semantic Branch)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
