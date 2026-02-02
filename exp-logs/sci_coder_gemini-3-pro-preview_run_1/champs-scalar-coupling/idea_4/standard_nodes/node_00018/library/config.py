import os
import random
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Global Configuration & Directories
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = "./submission"

# Ensure necessary writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "graph_cache"), exist_ok=True)

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
# Metadata (Split definitions)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data
STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Files (Parquet/NPY)
TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
GRAPH_CACHE_DIR = os.path.join(WORKING_DIR, "graph_cache")
FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Physical & Domain Constants
# -----------------------------------------------------------------------------
# Bond Length Threshold (Angstroms)
# Used to construct the molecular graph. 1.6A covers C-C, C-N, C-O, C-H bonds.
BOND_LENGTH_THRESHOLD = 1.6

# Coupling Types to Predict
COUPLING_TYPES = ["1JHC", "2JHH", "1JHN", "2JHN", "2JHC", "3JHC", "3JHH", "3JHN"]

# Atom Types (for context features)
ATOM_TYPES = ["H", "C", "N", "O", "F"]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# XGBoost Parameters for Path-Contextualized Stratified Ensemble
# Designed for high-capacity learning of geometric features (angles, dihedrals)
XGB_PARAMS = {
    "n_estimators": 30000,  # High ceiling, relying on early stopping
    "learning_rate": 0.02,  # Low LR for robust convergence
    "max_depth": 11,  # Deep trees to capture high-order geometric interactions
    "colsample_bytree": 0.45,  # Aggressive sampling to force diversity in split features
    "subsample": 0.8,  # Row subsampling for generalization
    "reg_alpha": 0.1,  # L1 Regularization
    "reg_lambda": 1.0,  # L2 Regularization
    "min_child_weight": 1,
    "n_jobs": 12,  # CPU threads
    "device": "cuda",  # GPU Acceleration (NVIDIA A100)
    "tree_method": "hist",  # Histogram-based method
    "random_state": RANDOM_STATE,
    "eval_metric": "mae",  # Metric: Mean Absolute Error
    "verbose": 0,  # Silent mode
}

# Training Control
EARLY_STOPPING_ROUNDS = 100


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
