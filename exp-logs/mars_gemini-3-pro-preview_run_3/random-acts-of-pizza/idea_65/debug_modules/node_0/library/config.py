import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_65"
SUBMISSION_DIR = "./submission"

# Subdirectories for artifacts
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata File Paths (Pre-stratified)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# Debug Mode: Set to True to speed up pipeline for testing
DEBUG = False

if DEBUG:
    N_FOLDS = 2
    N_ESTIMATORS_BOOST = 10
    N_ESTIMATORS_BAG = 10
else:
    N_FOLDS = 5
    N_ESTIMATORS_BOOST = 1000
    N_ESTIMATORS_BAG = 300

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
# Text Columns for Concatenation
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Community/History Column
SUBREDDIT_COL = "requester_subreddits_at_request"

# Augmented Global Metadata Allow-List
# Strictly excludes leakage (columns ending in _at_retrieval)
# Includes restored RAOP history and UTC timestamps
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
# VECTORIZER HYPERPARAMETERS
# =============================================================================
# Granular Lexical Vectorizer (Text)
LEXICAL_VECTORIZER_PARAMS = {
    "min_df": 2,
    "token_pattern": r"\w{1,}",  # Capture single characters like 'I', '$'
    "sublinear_tf": True,
    "ngram_range": (1, 2),
    "max_features": 20000,
    "strip_accents": "unicode",
}

# Community Vectorizer (History)
COMMUNITY_VECTORIZER_PARAMS = {
    "max_features": 1000,  # Top 1000 subreddits
    "binary": True,  # Bag-of-Concepts approach
    "token_pattern": r"(?u)\b\w\w+\b",
    "strip_accents": "unicode",
}

# =============================================================================
# MODEL HYPERPARAMETERS (LEVEL 1: BASE LEARNERS)
# =============================================================================

# --- Branch 1: Sparse Lexical (Text) ---
LEXICAL_BAGGER_PARAMS = {
    "n_estimators": N_ESTIMATORS_BAG,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

LEXICAL_ANCHOR_PARAMS = {
    "C": 1.0,
    "solver": "saga",
    "penalty": "l2",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# --- Branch 2: Sparse Behavioral (History) ---
COMMUNITY_BAGGER_PARAMS = {
    "n_estimators": 200 if not DEBUG else 10,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

COMMUNITY_ANCHOR_PARAMS = {
    "C": 1.0,
    "solver": "saga",
    "penalty": "l2",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# --- Branch 3: Dense Semantic (Text Embeddings) ---
# Conservative Boosting for XGB
SEMANTIC_BOOSTER_PARAMS = {
    "n_estimators": N_ESTIMATORS_BOOST,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "scale_pos_weight": 3.0,  # Approx ratio 75/25
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
    # Trainer should handle early_stopping_rounds via fit params
}

# Diversity via LightGBM
SEMANTIC_GRADIENT_PARAMS = {
    "n_estimators": N_ESTIMATORS_BOOST,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "is_unbalance": True,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

# Structural Diversity via RF
SEMANTIC_BAGGER_PARAMS = {
    "n_estimators": N_ESTIMATORS_BAG,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# --- Branch 4: Contextual (Metadata) ---
METADATA_ANCHOR_PARAMS = {
    "C": 0.5,
    "solver": "liblinear",
    "penalty": "l1",  # L1 for explicit feature selection
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

TEMPORAL_BOOSTER_PARAMS = {
    "n_estimators": 500 if not DEBUG else 10,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "is_unbalance": True,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

# =============================================================================
# MODEL HYPERPARAMETERS (LEVEL 2: META LEARNER)
# =============================================================================
META_LEARNER_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "penalty": "l2",
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
}

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
# List of all base models to instantiate
MODEL_KEYS = [
    "lexical_bagger",
    "lexical_anchor",
    "community_bagger",
    "community_anchor",
    "semantic_booster",
    "semantic_gradient",
    "semantic_bagger",
    "metadata_anchor",
    "temporal_booster",
]

# Classification for Hybrid Inference Protocol
# Volatile: Use CV-Bagging (Average of K-Fold models)
# Stable: Use Full Retraining (Train on Train+Val)
VOLATILE_MODELS = ["semantic_booster", "semantic_gradient", "temporal_booster"]

STABLE_MODELS = [
    "lexical_bagger",
    "lexical_anchor",
    "community_bagger",
    "community_anchor",
    "semantic_bagger",
    "metadata_anchor",
]

# Early stopping rounds for volatile models
EARLY_STOPPING_ROUNDS = 50
