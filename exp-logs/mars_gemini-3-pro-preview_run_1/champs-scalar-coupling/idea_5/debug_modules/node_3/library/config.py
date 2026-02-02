import os

# -----------------------------------------------------------------------------
# Directory and File Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Files
STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Pre-split)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# -----------------------------------------------------------------------------
# Physics and Chemistry Constants
# -----------------------------------------------------------------------------
# Covalent Radii (Angstroms) - Used for Adaptive Graph Construction
# Values based on standard Pyykko or Cordero radii
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}

# Pauling Electronegativity - Used for Level 0 Atom Features
ATOM_ELECTRONEGATIVITY = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98}

# Atomic Numbers
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

# Bond Tolerance (Angstroms)
# Connection exists if dist < r_i + r_j + BOND_TOLERANCE
BOND_TOLERANCE = 0.3

# Coupling Types for Stratification
COUPLING_TYPES = ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

# XGBoost Hyperparameters
# Designed for high capacity (deep trees) with regularization (colsample)
# to leverage high-dimensional topological features.
XGB_PARAMS = {
    "n_estimators": 50000,  # High ceiling, controlled by early stopping
    "learning_rate": 0.01,  # Low LR for better convergence
    "max_depth": 11,  # Deep trees (10-12 range) for complex interactions
    "colsample_bytree": 0.45,  # 0.4-0.5 to force usage of diverse features
    "subsample": 0.8,  # Prevent overfitting
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
    "min_child_weight": 1,
    "n_jobs": 12,  # CPU threads
    "device": "cuda",  # GPU acceleration
    "tree_method": "hist",  # Efficient histogram-based method
    "random_state": RANDOM_STATE,
    "eval_metric": "mae",  # Optimize Mean Absolute Error
}

# Training Configuration
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 1000
