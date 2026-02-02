import os
import random
import numpy as np
import torch

# ==========================================
# Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_35"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Hyperparameters and Constants
# ==========================================
SEED = 42
EPSILON = 1e-15

# OAS Estimator Parameters
# assume_centered=True is critical for geometric consistency when
# calculating covariance on residuals (X - mu).
OAS_PARAMS = {"assume_centered": True}


# ==========================================
# Utility Functions
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_alphanumeric_feature_order():
    """
    Generates the list of feature column names and sorts them alphanumerically.

    This enforces the specific memory layout:
    ['margin1', 'margin10', 'margin11', ..., 'margin2', ...]

    This layout is required to replicate the floating-point associativity
    conditions of the high-performance baseline.
    """
    feature_types = ["margin", "shape", "texture"]
    features = []

    # Generate all 192 feature names
    for ft in feature_types:
        for i in range(1, 65):
            features.append(f"{ft}{i}")

    # Return sorted alphanumerically (default string sort)
    return sorted(features)
