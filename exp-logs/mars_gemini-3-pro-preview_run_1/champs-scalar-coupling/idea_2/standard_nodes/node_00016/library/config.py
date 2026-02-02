import os

# -----------------------------------------------------------------------------
# Global Directories and Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# Atomic Covalent Radii (in Angstroms)
# Used for bond calculation and geometric feature engineering
ATOMIC_RADII = {"H": 0.38, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71}

# List of all scalar coupling types to predict
COUPLING_TYPES = ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]

# Atom types present in the dataset
ATOM_TYPES = ["H", "C", "N", "O", "F"]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# XGBoost parameters optimized for stratified training
XGB_PARAMS = {
    "n_estimators": 60000,  # Increased capacity to prevent premature stopping (Cite solution_lesson_node_00013)
    "learning_rate": 0.01,  # Reduced for precise convergence on full data (Cite solution_lesson_node_00008)
    "colsample_bytree": 0.5,  # Reduced to prevent overfitting after removing atom-type features
    "objective": "reg:absoluteerror",
    "n_jobs": 12,  # Utilizing available vCPUs
    "random_state": RANDOM_SEED,
    "tree_method": "hist",  # Efficient histogram-based algorithm
    "device": "cuda",  # Enable GPU acceleration
    "subsample": 0.7,  # Standard regularization
    "max_depth": 9,  # Sufficient depth for complex geometric interactions
}
