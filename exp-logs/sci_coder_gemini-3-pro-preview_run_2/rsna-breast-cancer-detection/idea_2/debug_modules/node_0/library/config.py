import os
import random
import numpy as np
import torch

# ====================================================
# Path Configuration
# ====================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# ====================================================
# Reproducibility Configuration
# ====================================================
SEED = 42

# ====================================================
# Data Configuration
# ====================================================
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 12

# ====================================================
# Stage 1: Feature Extractor (Backbone) Configuration
# ====================================================
BACKBONE_NAME = "resnet18"
EMBEDDING_SIZE = 512  # Output dimension of ResNet18 Global Average Pooling

# ====================================================
# Stage 2: Classifier (LightGBM) Configuration
# ====================================================
# Parameters for LightGBM
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "is_unbalance": True,  # Handle the ~2% positive class imbalance
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": SEED,
}

# Training loop parameters
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100

# ====================================================
# Hardware Configuration
# ====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_system(seed=SEED):
    """
    Initializes the environment for reproducibility and ensures necessary
    directories exist.

    Args:
        seed (int): The random seed to use for all libraries.
    """
    # Create necessary directories
    for directory in [WORKING_DIR, SUBMISSION_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variables for determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
