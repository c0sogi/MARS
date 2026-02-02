import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Raw Input Files
STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
DIPOLE_PATH = os.path.join(INPUT_DIR, "dipole_moments.csv")
POTENTIAL_PATH = os.path.join(INPUT_DIR, "potential_energy.csv")
MULLIKEN_PATH = os.path.join(INPUT_DIR, "mulliken_charges.csv")
MAGNETIC_PATH = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
CONTRIBUTIONS_PATH = os.path.join(INPUT_DIR, "scalar_coupling_contributions.csv")

# Metadata Files (Pre-split)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# PHYSICS & CHEMISTRY CONSTANTS
# =============================================================================
# Covalent Radii (Angstroms) for Adaptive Thresholding
# Source: Alvarez (2008) or similar standard tables
ATOM_RADII = {"H": 0.38, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71}

# Tolerance added to sum of radii to determine connectivity (Angstroms)
# Bond exists if dist < r_i + r_j + BOND_TOLERANCE
BOND_TOLERANCE = 0.3

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
N_JOBS = 12

# Debugging / Development Controls
# Set DEBUG to True to run on a small subset of data for testing pipeline
DEBUG = False
DEBUG_SAMPLE_SIZE = 5000  # Number of molecules to use in debug mode

# =============================================================================
# MODEL HYPERPARAMETERS (XGBoost)
# =============================================================================
# Base parameters for XGBoost Regressor
# Optimized for NVIDIA A100 GPU usage
BASE_XGB_PARAMS = {
    "booster": "gbtree",
    "device": "cuda",  # Use GPU acceleration
    "tree_method": "hist",  # Histogram-based algorithm for GPU
    "objective": "reg:absoluteerror",  # Optimize MAE directly
    "eval_metric": "mae",
    "learning_rate": 0.01,  # Low learning rate for better convergence
    "n_estimators": 40000,  # High ceiling, controlled by early stopping
    "colsample_bytree": 0.4,  # Strong feature subsampling to force topology usage
    "subsample": 0.8,  # Row subsampling
    "reg_alpha": 0.1,  # L1 Regularization
    "reg_lambda": 1.0,  # L2 Regularization
    "n_jobs": N_JOBS,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

# Type-Specific Parameters
# We use a Stratified Ensemble approach.
# All types use deep trees (max_depth=12) to capture high-order interactions
# as per the VH-FASE strategy.
TYPE_SPECIFIC_PARAMS = {
    "1JHC": {**BASE_XGB_PARAMS, "max_depth": 12},
    "1JHN": {**BASE_XGB_PARAMS, "max_depth": 12},
    "2JHC": {**BASE_XGB_PARAMS, "max_depth": 12},
    "2JHH": {**BASE_XGB_PARAMS, "max_depth": 12},
    "2JHN": {**BASE_XGB_PARAMS, "max_depth": 12},
    "3JHC": {**BASE_XGB_PARAMS, "max_depth": 12},
    "3JHH": {**BASE_XGB_PARAMS, "max_depth": 12},
    "3JHN": {**BASE_XGB_PARAMS, "max_depth": 12},
}

# Training Control
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 500
