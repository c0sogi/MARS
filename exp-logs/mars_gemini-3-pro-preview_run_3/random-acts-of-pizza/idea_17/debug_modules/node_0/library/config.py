import os

# =============================================================================
# GLOBAL SETTINGS & REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42
N_FOLDS = 5
DEBUG_SAMPLE_SIZE = (
    None  # Set to an integer (e.g., 100) for debugging, None for full run
)

# =============================================================================
# FILE PATHS
# =============================================================================
# Input Metadata (Pre-generated)
TRAIN_DATA_PATH = "./metadata/train.parquet"
VAL_DATA_PATH = "./metadata/val.parquet"
TEST_DATA_PATH = "./metadata/test.parquet"

# Working Directory for Caching Intermediate Features (Idea 17)
CACHE_DIR = "./working/idea_17/"
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission Output
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA SCHEMA
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"
TEXT_COL = "request_text_edit_aware"  # Use edit-aware text to prevent leakage
TITLE_COL = "request_title"
SUBREDDIT_COL = "requester_subreddits_at_request"
USER_COL = "requester_username"

# Columns containing this suffix will be dropped to prevent data leakage from the future
LEAKAGE_SUFFIX = "_at_retrieval"

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================
# Semantic Embedding Model (Sentence-Transformers)
TRANSFORMER_MODEL_NAME = "all-mpnet-base-v2"

# TF-IDF Vectorization Settings
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_SUBLINEAR_TF = True

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Random Forest (Base Learner - Sparse & Dense Views)
# Configured with high regularization and balanced class weights
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# XGBoost (Base Learner - Dense Views)
# Configured for imbalanced data; early stopping to be handled in training loop
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": 3.0,  # Approximate ratio of Negative/Positive class
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}
XGB_EARLY_STOPPING_ROUNDS = 50

# Logistic Regression (Base Learner - Contextual Anchor)
LOGREG_ANCHOR_PARAMS = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

# Stacking Meta-Learner
META_LEARNER_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}
