import os

# =============================================================================
# 1. File System Paths
# =============================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea to avoid file conflicts
WORKING_DIR = "./working/idea_12"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Generated previously)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Feature Cache Paths
# These paths will be used by the feature extraction module to save/load processed data
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Final Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# 2. Feature Extraction Hyperparameters
# =============================================================================

# Radial Distribution Function (RDF) Settings
# Captures 2-body distance distributions
RDF_BINS = 40
RDF_MIN_DIST = 0.0
RDF_MAX_DIST = 10.0  # Angstroms

# Angular Distribution Function (ADF) Settings
# Captures 3-body bond angle distributions
ADF_BINS = 40
ADF_MIN_ANGLE = 0.0
ADF_MAX_ANGLE = 180.0  # Degrees

# Neighbor Cutoff
# Defines the maximum distance between atoms to consider them "bonded" or neighbors
# for the purpose of calculating angles. 3.0 Angstroms covers typical metal-oxide bonds.
CUTOFF_DISTANCE = 3.0

# =============================================================================
# 3. Model Hyperparameters (XGBoost)
# =============================================================================

RANDOM_SEED = 42

# XGBoost Regressor Parameters
# Tuned for generalization: low learning rate, high estimators, subsampling enabled
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient training method
}

# =============================================================================
# 4. Runtime / Debugging Controls
# =============================================================================

# Set to True to run on a small subset of data for quick code verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 50
