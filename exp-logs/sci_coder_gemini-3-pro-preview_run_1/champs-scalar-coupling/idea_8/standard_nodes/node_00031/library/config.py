import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
# Input Data
INPUT_DIR = "./input"
STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")
STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")

# Metadata (Pre-generated splits)
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Idea 8 (Cache & Models)
WORKING_DIR = "./working/idea_8"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "xgb_models")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# PHYSICS & CHEMISTRY CONSTANTS
# =============================================================================
# Covalent Radii (Angstroms) - Used for adaptive connectivity thresholds
# Values approx based on Alvarez (2008)
ATOMIC_RADII = {"H": 0.38, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71}

# Atomic Numbers for feature encoding
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

# List of all scalar coupling types
COUPLING_TYPES = ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]

# =============================================================================
# COMPUTATIONAL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
N_JOBS = 12  # Available vCPUs

# =============================================================================
# MODEL HYPERPARAMETERS (XGBoost)
# =============================================================================
# High-capacity configuration as per "Lesson 00017" and "Lesson 00013"
# - Deep trees (max_depth 10-12) to capture high-order geometric interactions
# - Low learning rate (0.01) with high estimators for convergence
# - Colsample_bytree (0.4) to force usage of diverse topological features

XGB_PARAMS = {
    "common": {
        "booster": "gbtree",
        "tree_method": "hist",  # Faster on large data
        "device": "cuda",  # Use NVIDIA A100
        "objective": "reg:absoluteerror",  # Metric is MAE
        "eval_metric": "mae",
        "learning_rate": 0.01,
        "max_depth": 10,  # Deep trees for interaction capture
        "subsample": 0.8,
        "colsample_bytree": 0.4,  # Regularization to prevent overfitting to distance
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": N_JOBS,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    },
    "training": {
        "n_estimators": 20000,  # High ceiling, rely on early stopping
        "early_stopping_rounds": 100,
        "verbose": 1000,  # Print progress every 1000 rounds
    },
}

# Specific overrides per type if necessary (currently using common high-capacity config)
# Can be extended if specific types need different regularization
TYPE_SPECIFIC_PARAMS = {
    "1JHC": {"max_depth": 12},  # Most complex/abundant type
    "2JHH": {"max_depth": 10},
    # Others inherit common
}


def get_xgb_params(coupling_type):
    """Returns the merged parameter dictionary for a specific coupling type."""
    params = XGB_PARAMS["common"].copy()
    if coupling_type in TYPE_SPECIFIC_PARAMS:
        params.update(TYPE_SPECIFIC_PARAMS[coupling_type])
    return params
