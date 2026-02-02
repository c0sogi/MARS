import os
import numpy as np

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Reproducibility
SEED = 42
np.random.seed(SEED)

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA DEFINITIONS
# =============================================================================

# Key Columns
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"
TEXT_COL = "request_text_edit_aware"  # Use edit-aware text to prevent leakage
TITLE_COL = "request_title"
SUBREDDIT_COL = "requester_subreddits_at_request"

# Columns to Exclude
# 1. Identifiers and Raw Text (processed separately)
# 2. Leakage Features (available only at retrieval time)
EXCLUDE_COLS = [
    "request_id",
    "requester_username",
    "giver_username_if_known",
    "source_file",
    "request_text",
    "request_text_edit_aware",
    "request_title",
    "requester_subreddits_at_request",
    "requester_user_flair",  # Often updated after success
    "post_was_edited",  # Often updated after success
    # Retrieval-time stats (LEAKAGE)
    "number_of_downvotes_of_request_at_retrieval",
    "number_of_upvotes_of_request_at_retrieval",
    "request_number_of_comments_at_retrieval",
    "requester_account_age_in_days_at_retrieval",
    "requester_days_since_first_post_on_raop_at_retrieval",
    "requester_number_of_comments_at_retrieval",
    "requester_number_of_comments_in_raop_at_retrieval",
    "requester_number_of_posts_at_retrieval",
    "requester_number_of_posts_on_raop_at_retrieval",
    "requester_upvotes_minus_downvotes_at_retrieval",
    "requester_upvotes_plus_downvotes_at_retrieval",
]

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================

# Text Processing
TFIDF_PARAMS = {
    "max_features": 3000,
    "stop_words": "english",
    "sublinear_tf": True,
    "min_df": 5,
    "ngram_range": (1, 2),
}

# Dense Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Manifold Learning
PCA_COMPONENTS = 50

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Training Strategy
N_FOLDS = 5

# 1. Random Forest (Used for Lexical, Behavioral, and Semantic Bagging)
# Regularized with min_samples_leaf=2 to prevent overfitting on sparse data
RF_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 2. XGBoost (Used for Semantic Boosting)
# scale_pos_weight set to approx 3.0 (75% neg / 25% pos)
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "eval_metric": "logloss",
    # Early stopping rounds will be passed dynamically in training loop
}

# 3. k-Nearest Neighbors (Used for Manifold Branch)
KNN_PARAMS = {
    "n_neighbors": 20,
    "weights": "distance",
    "metric": "cosine",
    "n_jobs": -1,
}

# 4. Logistic Regression (Used for Contextual Branch & Meta-Learner)
LR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": SEED,
}
