import os

# ==========================================
# DIRECTORY AND FILE PATHS
# ==========================================
# Base input directory
INPUT_DIR = "./input"

# Metadata paths (pre-generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for intermediate files and caching
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_optimized")

# Submission directory
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# DATA CONFIGURATION
# ==========================================
# Reproducibility
RANDOM_SEED = 42

# Data Structure
NUM_SENSORS = 10
SENSOR_COLS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]

# Feature Engineering: List of statistics to compute per sensor
# These correspond to the aggregation functions applied to the time-series data
STATS_COLS = [
    "mean",
    "std",
    "min",
    "max",
    "skew",
    "kurt",
    "q01",  # 1st percentile
    "q05",  # 5th percentile
    "q25",  # 25th percentile
    "q50",  # Median
    "q75",  # 75th percentile
    "q95",  # 95th percentile
    "q99",  # 99th percentile
    "mad",  # Mean Absolute Deviation
]

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
# LightGBM Parameters optimized for MAE regression
LGBM_PARAMS = {
    "objective": "regression_l1",  # L1 loss (Mean Absolute Error)
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "n_jobs": 12,  # Utilizing available vCPUs
    "seed": RANDOM_SEED,
    "force_col_wise": True,
}

# Training Loop Configuration
NUM_FOLDS = 5
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

# ==========================================
# SYSTEM RESOURCES
# ==========================================
N_JOBS = 12
