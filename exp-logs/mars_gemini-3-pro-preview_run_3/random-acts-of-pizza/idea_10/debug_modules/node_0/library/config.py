import os

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# Directories & Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_10"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Data Column Definitions
# -----------------------------------------------------------------------------
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"
TEXT_COL = "request_text_edit_aware"
TITLE_COL = "request_title"
SUBREDDIT_COL = "requester_subreddits_at_request"
USER_COL = "requester_username"

# Columns to filter out during feature selection (IDs, Target, Raw Text, etc.)
# Note: Retrieval columns should be dynamically dropped based on suffix
DROP_COLS = [
    ID_COL,
    TARGET_COL,
    "request_text",
    "request_text_edit_aware",
    "request_title",
    "requester_subreddits_at_request",
    "requester_username",
    "source_file",
    "giver_username_if_known",
    "requester_user_flair",
    "post_was_edited",
]

RETRIEVAL_SUFFIX = "_at_retrieval"

# -----------------------------------------------------------------------------
# Feature Engineering Hyperparameters
# -----------------------------------------------------------------------------

# Lexical View (Text TF-IDF)
TEXT_TFIDF_PARAMS = {
    "max_features": 3000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "sublinear_tf": True,
}

# Behavioral View (Subreddit TF-IDF)
SUBREDDIT_TFIDF_PARAMS = {
    "max_features": 1000,
    "ngram_range": (1, 1),
    "stop_words": "english",
    "sublinear_tf": True,
}

# Behavioral View (Subreddit SVD)
SUBREDDIT_SVD_COMPONENTS = 20

# Semantic View (SBERT)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------

# Level 1: Lexical Bagger (Random Forest)
RF_LEXICAL_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# Level 1: Behavioral Bagger (Random Forest)
RF_BEHAVIORAL_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# Level 1: Contextual Booster (XGBoost)
XGB_CONTEXTUAL_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

XGB_FIT_PARAMS = {"early_stopping_rounds": 50, "verbose": False}

# Level 2: Meta-Learner (Logistic Regression)
STACKING_META_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": SEED,
    "class_weight": None,  # Stacking usually handles calibration without forcing weights
}

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
NUM_FOLDS = 5
