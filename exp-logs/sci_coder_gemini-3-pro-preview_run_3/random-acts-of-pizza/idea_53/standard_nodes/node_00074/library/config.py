import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_53"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_SEED = 42
N_FOLDS = 5
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# =============================================================================
# FEATURE COLUMNS
# =============================================================================

# 1. Text Columns for Concatenation (Title + Edit-Aware Body)
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# 2. Subreddit History Column
SUBREDDIT_COL = "requester_subreddits_at_request"

# 3. Metadata Allow-List (Hygienic Feature Selection)
# Includes restored RAOP history features and raw timestamps
METADATA_FEATURES = [
    "unix_timestamp_of_request_utc",
    "requester_account_age_in_days_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
    # Restored RAOP History Features
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_days_since_first_post_on_raop_at_request",
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. LEXICAL BRANCH (Text Modality - Sparse)
# -----------------------------------------------------------------------------

# Lexical Bagger (Random Forest on TF-IDF)
LEXICAL_BAGGER_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,  # Regularization to prevent overfitting
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# Lexical Anchor (Logistic Regression on TF-IDF)
LEXICAL_ANCHOR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
}

# TF-IDF Vectorizer Settings for Lexical Branch
LEXICAL_VECTORIZER_PARAMS = {
    "strip_accents": "unicode",
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 5,
    "sublinear_tf": True,
    "max_features": 10000,
}

# -----------------------------------------------------------------------------
# 2. BEHAVIORAL BRANCH (History Modality - Sparse)
# -----------------------------------------------------------------------------

# Community Bagger (Random Forest on Subreddit History)
COMMUNITY_BAGGER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# Vectorizer for Subreddits (Bag-of-Concepts)
COMMUNITY_VECTORIZER_PARAMS = {
    "max_features": 1000,  # Strict vocabulary limit
    "binary": True,  # Presence/Absence
    "token_pattern": r"(?u)\b\w+\b",  # Simple tokenization for subreddit names
}

# -----------------------------------------------------------------------------
# 3. SEMANTIC BRANCH (Text Modality - Dense)
# -----------------------------------------------------------------------------
# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Semantic Booster (XGBoost on Embeddings + Meta)
SEMANTIC_BOOSTER_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,  # Handle imbalance
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "verbosity": 0,
}

# Semantic Gradient (LightGBM on Embeddings + Meta)
SEMANTIC_GRADIENT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
    "early_stopping_rounds": 50,
}

# Semantic Bagger (Random Forest on Embeddings + Meta)
SEMANTIC_BAGGER_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,  # Modality-specific regularization
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# -----------------------------------------------------------------------------
# 4. CONTEXTUAL BRANCH (Metadata Modality)
# -----------------------------------------------------------------------------

# Metadata Anchor (Logistic Regression on Meta)
METADATA_ANCHOR_PARAMS = {
    "C": 0.1,  # Stronger regularization
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

# Temporal Booster (LightGBM on Meta)
TEMPORAL_BOOSTER_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
    "early_stopping_rounds": 50,
}

# -----------------------------------------------------------------------------
# 5. META-LEARNER (Level 2)
# -----------------------------------------------------------------------------

META_LEARNER_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
}

# =============================================================================
# INFERENCE PROTOCOL CONFIG
# =============================================================================

# Defines which models are "Volatile" (require CV-Bagging/Early Stopping)
# vs "Stable" (can be retrained on full data).
MODEL_TYPES = {
    "lexical_bagger": "stable",
    "lexical_anchor": "stable",
    "community_bagger": "stable",
    "semantic_booster": "volatile",
    "semantic_gradient": "volatile",
    "semantic_bagger": "stable",
    "metadata_anchor": "stable",
    "temporal_booster": "volatile",
}
