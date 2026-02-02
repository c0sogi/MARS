import os

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------

SEED = 42

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_54")
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------

# Target Column
TARGET_COL = "requester_received_pizza"

# Text Columns to be concatenated for lexical/semantic analysis
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Community Column
COMMUNITY_COL = "requester_subreddits_at_request"

# Metadata Allow List (Hygienic Feature Selection)
# Includes restored RAOP history features as per Idea 54
ALLOW_LIST_METADATA = [
    # Temporal
    "unix_timestamp_of_request_utc",
    # User Stats (General)
    "requester_account_age_in_days_at_request",
    "requester_upvotes_minus_downvotes_at_request",  # Karma
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    # User Stats (RAOP Specific - Restored Priors)
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_days_since_first_post_on_raop_at_request",
]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------

# Common settings
N_FOLDS = 5
N_JOBS = 12  # Using available vCPUs

# 1. Sparse Lexical Branch
# Lexical Bagger (Random Forest)
LEXICAL_BAGGER_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,  # Regularization
    "max_features": "sqrt",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "class_weight": "balanced",
}

# Lexical Anchor (Logistic Regression)
LEXICAL_ANCHOR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",  # Good for high dimensional sparse data
    "random_state": SEED,
    "class_weight": "balanced",
}

# 2. Sparse Behavioral Branch
# Community Bagger (Random Forest)
COMMUNITY_BAGGER_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 3. Dense Semantic Branch
# Semantic Booster (XGBoost)
# Note: scale_pos_weight handles imbalance. Approx ratio 3:1 (0.75/0.25) -> ~3.0
SEMANTIC_BOOSTER_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,  # Handling imbalance
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "verbosity": 0,
}

# Semantic Gradient (LightGBM)
SEMANTIC_GRADIENT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": -1,
    "early_stopping_rounds": 50,
}

# Semantic Bagger (Random Forest)
SEMANTIC_BAGGER_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,  # Modality-Specific Regularization
    "min_samples_leaf": 4,  # Modality-Specific Regularization
    "max_features": "sqrt",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 4. Contextual Branch
# Metadata Anchor (Logistic Regression)
METADATA_ANCHOR_PARAMS = {
    "C": 0.1,  # Stronger regularization for smaller feature set
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": SEED,
    "class_weight": "balanced",
}

# Temporal Booster (LightGBM)
TEMPORAL_BOOSTER_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,  # Smaller model for fewer features
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": -1,
    "early_stopping_rounds": 50,
}

# Level 2 Meta-Learner (Logistic Regression)
META_LEARNER_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": SEED,
}

# -----------------------------------------------------------------------------
# Feature Extraction Config
# -----------------------------------------------------------------------------

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "sublinear_tf": True,
    "max_features": 20000,
    "stop_words": "english",
}

COMMUNITY_VOCAB_SIZE = 1000
