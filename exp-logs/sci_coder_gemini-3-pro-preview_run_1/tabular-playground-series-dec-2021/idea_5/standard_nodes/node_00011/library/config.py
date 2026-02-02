import os

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Cache directory for idea_5 as specified in the strategy
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Raw Data Paths (Full Dataset)
RAW_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
RAW_TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Paths (Pre-split 80/20)
METADATA_TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
METADATA_VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
METADATA_TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
N_FOLDS = 5

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# XGBoost parameters optimized for high capacity and ensemble diversity
XGB_PARAMS = {
    "max_depth": 10,
    "learning_rate": 0.05,  # eta
    "tree_method": "hist",
    "device": "cuda",  # GPU acceleration
    "subsample": 0.8,  # Row subsampling for diversity
    "colsample_bytree": 0.8,  # Feature subsampling for diversity
    "objective": "multi:softprob",  # Output probabilities for soft voting
    "eval_metric": "mlogloss",
    "num_class": 8,  # Labels are 1-7, so we need size 8 (0-7)
    "n_jobs": 12,
    "random_state": SEED,
    "verbosity": 0,
}

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50
