import os

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_61"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODELS_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
N_FOLDS = 5
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# Text Columns
TEXT_COLS = ["request_title", "request_text_edit_aware"]
SUBREDDIT_COL = "requester_subreddits_at_request"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# Explicit Allow-List for Augmented Global Metadata
# We strictly include only pre-request signals and restored RAOP history
METADATA_ALLOWLIST = [
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

# Columns to strictly exclude (Leakage Prevention)
RETRIEVAL_EXCLUDE_SUFFIX = "_at_retrieval"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Sparse Lexical Branch (Text Modality)
# Granular Lexical Bagger: RF on TF-IDF (Title + Body)
LEXICAL_VECTORIZER_PARAMS = {
    "sublinear_tf": True,
    "min_df": 5,
    "ngram_range": (1, 2),
    "token_pattern": r"\w{1,}",  # Capture single characters like "I", "$"
    "stop_words": "english",
}
LEXICAL_BAGGER_PARAMS = {
    "n_estimators": 100,
    "min_samples_leaf": 2,  # Regularization
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 2. Sparse Behavioral Branch (History Modality)
# Community Bagger: RF on TF-IDF (Subreddits)
COMMUNITY_VECTORIZER_PARAMS = {
    "max_features": 1000,  # Top 1000 vocabulary limit
    "token_pattern": r"(?u)\b\w+\b",
}
COMMUNITY_BAGGER_PARAMS = {
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 3. Dense Semantic Branch (Text Modality)
# Semantic Booster: XGBoost on Embeddings
SEMANTIC_BOOSTER_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,  # Conservative boosting
    "colsample_bytree": 0.6,  # Feature subsampling
    "subsample": 0.8,
    "max_depth": 6,
    "n_jobs": -1,
    "random_state": SEED,
    "enable_categorical": False,
    # scale_pos_weight should be calculated dynamically based on train set imbalance
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
}

# Semantic Gradient: LightGBM on Embeddings
SEMANTIC_GRADIENT_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
    "metric": "auc",
    "early_stopping_round": 100,
    "class_weight": "balanced",
}

# Semantic Bagger: Random Forest on Embeddings
SEMANTIC_BAGGER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,  # Modality-Specific Regularization
    "min_samples_leaf": 4,
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced",
}

# 4. Contextual Branch (Metadata Modality)
# Metadata Anchor: Logistic Regression on Metadata
METADATA_ANCHOR_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": SEED,
    "class_weight": "balanced",
}

# Temporal Booster: LightGBM on Metadata
TEMPORAL_BOOSTER_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 15,  # Smaller capacity for fewer features
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
    "metric": "auc",
    "early_stopping_round": 50,
    "class_weight": "balanced",
}

# =============================================================================
# LEVEL 2 META-LEARNER
# =============================================================================
META_LEARNER_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "random_state": SEED,
    # No class_weight here as inputs are probabilities
}
