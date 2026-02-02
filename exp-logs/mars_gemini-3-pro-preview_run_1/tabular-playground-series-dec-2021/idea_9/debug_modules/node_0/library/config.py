import os

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific working directory for this experimental iteration (Idea 9)
IDEA_DIR = os.path.join(WORKING_DIR, "idea_9")
CACHE_DIR = IDEA_DIR

# Ensure necessary directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Dataset Paths (using metadata splits)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Data Configuration
# =============================================================================
ID_COL = "Id"
TARGET_COL = "Cover_Type"
SEED = 42

# Class Mapping
# The dataset contains classes [1, 2, 3, 4, 6, 7]. Class 5 is missing.
# XGBoost requires targets to be integers in [0, num_class-1].
TARGET_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
INVERSE_TARGET_MAPPING = {v: k for k, v in TARGET_MAPPING.items()}
NUM_CLASSES = len(TARGET_MAPPING)

# =============================================================================
# Model & Training Configuration
# =============================================================================
N_FOLDS = 5
PSEUDO_LABEL_THRESHOLD = 0.99
NUM_BOOST_ROUND = 3000  # High ceiling, controlled by Early Stopping
EARLY_STOPPING_ROUNDS = 50

# XGBoost Hyperparameters
# Optimized for GPU acceleration and high capacity (Deep Trees)
XGB_PARAMS = {
    "objective": "multi:softmax",
    "num_class": NUM_CLASSES,
    "eval_metric": "mlogloss",  # LogLoss for better probability calibration
    "tree_method": "hist",
    "device": "cuda",  # Use NVIDIA A100
    "max_depth": 10,  # High capacity
    "learning_rate": 0.05,  # Conservative learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": 12,  # vCPUs
    "verbosity": 0,
}
