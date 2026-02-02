import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_51"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# =============================================================================
# FEATURE ALLOW-LISTS (HYGIENIC FEATURE SELECTION)
# =============================================================================
# Explicitly selected metadata features to prevent leakage (no _at_retrieval)
# and capture raw temporal/behavioral signals.
METADATA_FEATURES = [
    # Raw Temporal Anchor
    "unix_timestamp_of_request_utc",
    # User Demographics / Stats
    "requester_account_age_in_days_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
    # RAOP Specific History (Valid Priors)
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_days_since_first_post_on_raop_at_request",
]

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# TF-IDF Configuration for Text (Sparse Lexical)
LEXICAL_VECTORIZER_PARAMS = {
    "analyzer": "word",
    "token_pattern": r"\w{1,}",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 5,
    "max_features": 20000,
    "stop_words": "english",
}

# TF-IDF Configuration for Subreddits (Sparse Behavioral)
# Strictly limited to top 1000 communities
COMMUNITY_VECTORIZER_PARAMS = {
    "analyzer": "word",
    "token_pattern": r"\w{1,}",
    "ngram_range": (1, 1),
    "max_features": 1000,
    "binary": True,  # Bag-of-Concepts approach
}

# =============================================================================
# HYPERPARAMETERS (HEPT-VIEW ENSEMBLE)
# =============================================================================
# Note: scale_pos_weight is approx ratio of neg/pos (0.75/0.25 = 3.0)

MODEL_PARAMS = {
    # 1. Sparse Lexical Branch
    "lexical_bagger": {
        "n_estimators": 200,
        "min_samples_leaf": 2,  # Regularization
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    },
    # 2. Sparse Behavioral Branch
    "community_bagger": {
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    },
    # 3. Dense Semantic Branch
    "semantic_booster": {  # XGBoost
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "scale_pos_weight": 3.0,
        "n_jobs": -1,
        "random_state": SEED,
        "verbosity": 0,
        "early_stopping_rounds": 100,
    },
    "semantic_gradient": {  # LightGBM
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "early_stopping_rounds": 100,
    },
    "semantic_bagger": {  # Random Forest
        "n_estimators": 200,
        "max_depth": 12,  # Modality-Specific Regularization
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    },
    # 4. Contextual Branch
    "metadata_anchor": {  # Logistic Regression
        "C": 0.1,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": SEED,
    },
    "temporal_booster": {  # LightGBM
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 15,  # Shallower for fewer features
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "early_stopping_rounds": 50,
    },
    # Level 2 Meta-Learner
    "meta_learner": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": SEED,
    },
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
N_FOLDS = 5
USE_HYBRID_INFERENCE = True  # Retrain stable learners on full data, bag volatile ones
