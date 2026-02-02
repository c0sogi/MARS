import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Input Data (using generated metadata for speed and consistency)
TRAIN_DATA_PATH = "./metadata/train.parquet"
VAL_DATA_PATH = "./metadata/val.parquet"
TEST_DATA_PATH = "./metadata/test.parquet"

# Working and Output Directories
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_8")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Final Submission File Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
N_FOLDS = 5
N_JOBS = 12  # Utilize available vCPUs

# =============================================================================
# DATA COLUMN CONFIGURATION
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"
TEXT_COL = "request_text_edit_aware"
TITLE_COL = "request_title"
SUBREDDIT_COL = "requester_subreddits_at_request"

# Columns to drop to prevent data leakage (retrieval-time features)
DROP_SUFFIXES = ["_at_retrieval"]

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# Lexical View: TF-IDF on Request Text
LEXICAL_VECTORIZER_PARAMS = {
    "max_features": 3000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "sublinear_tf": True,
    "min_df": 2,
    "max_df": 0.95,
}

# Behavioral View: TF-IDF on Subreddit History (Bag-of-Communities)
BEHAVIORAL_VECTORIZER_PARAMS = {
    "max_features": 1000,
    "ngram_range": (1, 1),
    "stop_words": None,
    "binary": True,  # Presence in a community is more significant than frequency
    "lowercase": True,
}

# Semantic View: Pre-trained Transformer
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Level 1: Base Learners (Random Forest)
# Applied to Lexical, Semantic, and Behavioral views independently
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,  # Allow deep trees for bagging
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbose": 0,
}

# Level 2: Meta Learner (Logistic Regression)
# Stacking Classifier to combine probabilities
LR_PARAMS = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": None,  # Probabilities are already calibrated by RF
    "random_state": SEED,
    "max_iter": 1000,
    "n_jobs": 1,
}
