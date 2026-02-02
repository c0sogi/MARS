import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_38")
SUBMISSION_DIR = "./submission"
MODEL_DIR = os.path.join(CACHE_DIR, "models")

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
SEED = 42
N_JOBS = 12  # Utilizing available vCPUs
N_FOLDS = 5  # For Nested Community Profiling and Stacking

# =============================================================================
# DATA DEFINITIONS
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"
TEXT_COLS = ["request_title", "request_text_edit_aware"]
SUBREDDIT_COL = "requester_subreddits_at_request"
TIMESTAMP_COL = "unix_timestamp_of_request_utc"

# Augmented Global Metadata (Allow-List)
# Note: The 'Community Generosity Score' will be generated dynamically and appended to this list during processing.
METADATA_COLS = [
    "unix_timestamp_of_request_utc",
    "requester_account_age_in_days_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_days_since_first_post_on_raop_at_request",
]

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# 1. Sparse Lexical Branch (Text Modality)
# TF-IDF on Concatenated Title + Body
TFIDF_TEXT_PARAMS = {
    "strip_accents": "unicode",
    "stop_words": "english",
    "ngram_range": (1, 2),
    "max_features": 10000,  # Capture high-impact keywords
    "sublinear_tf": True,
    "min_df": 5,
}

# 2. Sparse Behavioral Branch (History Modality)
# TF-IDF on Subreddit History (Bag-of-Concepts)
TFIDF_SUBREDDIT_PARAMS = {
    "strip_accents": "unicode",
    "stop_words": None,
    "lowercase": False,  # Subreddit names can be case-sensitive or standard
    "max_features": 1000,  # Strictly limit to Top 1,000 subreddits
    "binary": True,
}

# 3. Dense Semantic Branch (Text Modality)
# Frozen Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 384

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Level 1: Base Learners

# 1. Sparse Lexical Bagger (Random Forest)
RF_LEXICAL_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,  # Regularization
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": 0,
}

# 2. Sparse Behavioral Bagger (Random Forest)
RF_BEHAVIORAL_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": 0,
}

# 3. Dense Semantic Booster (XGBoost)
# Note: scale_pos_weight is calculated dynamically during training based on fold balance
XGB_SEMANTIC_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda",  # Utilize A100 GPU
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbosity": 0,
}
XGB_EARLY_STOPPING_ROUNDS = 50

# 4. Dense Semantic Bagger (Random Forest)
RF_SEMANTIC_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,  # Modality-Specific Regularization
    "min_samples_leaf": 4,  # Modality-Specific Regularization
    "class_weight": "balanced",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": 0,
}

# 5. Metadata Anchor (Logistic Regression)
LR_ANCHOR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 2000,
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

# Level 2: Meta-Learner

# Stacking Meta-Learner (Logistic Regression)
META_LEARNER_PARAMS = {
    "C": 0.1,  # Stronger regularization for meta-learner
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": None,  # Let the probabilities speak
    "random_state": SEED,
    "n_jobs": N_JOBS,
}
