import os
import random
import numpy as np
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Feature Cache Paths (Parquet format)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Configuration
# ==========================================
SEED = 42
NUM_SENSORS = 10
SENSOR_COLS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]

# Debugging / Sampling
# Can be used by data loaders to limit dataset size for rapid prototyping
DEBUG_SAMPLE_SIZE = 100

# ==========================================
# Model Configuration (LightGBM)
# ==========================================
# Hyperparameters optimized for MAE (L1 loss)
LGB_PARAMS = {
    "objective": "regression_l1",  # Minimizes Mean Absolute Error
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": SEED,
}

# Training Loop Configuration
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100


# ==========================================
# Utility Functions
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.

    Args:
        seed (int): The seed value to use. Defaults to global SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
