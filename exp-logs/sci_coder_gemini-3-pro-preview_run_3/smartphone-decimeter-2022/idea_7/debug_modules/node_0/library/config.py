import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
SUBMISSION_DIR = "./submission"

# Cache directory for this specific idea/experiment
# All intermediate processing steps (features, smoothed baselines) should be saved here
WORKING_DIR = "./working/idea_7"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# GLOBAL SEED
# =============================================================================
SEED = 42

# =============================================================================
# DATA CONTROL & DEBUGGING
# =============================================================================
# Set DEBUG to True to run on a small subset of data for rapid iteration
DEBUG = False

# Number of unique drives to sample if DEBUG is True
DEBUG_DRIVE_COUNT = 2

# Maximum number of rows to load/process if DEBUG is True (alternative to drive count)
MAX_DEBUG_ROWS = 10000

# =============================================================================
# MODEL HYPERPARAMETERS (LightGBM)
# =============================================================================
# Using 'regression_l1' (MAE) as the objective to be robust against heavy-tailed
# outliers in GNSS data, as per the strategy.
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}

# Training loop control
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

# =============================================================================
# KINEMATIC SMOOTHER CONFIGURATION
# =============================================================================
# Threshold for the Innovation Gate in the Kalman Smoother.
# Observations resulting in innovations (prediction errors) larger than this
# value (in meters) will be rejected to prevent smearing.
INNOVATION_THRESHOLD = 10.0
