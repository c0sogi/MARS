import os

# =============================================================================
# Global Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_20"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Configuration
# =============================================================================
RANDOM_SEED = 42
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Debugging: Set to an integer (e.g., 100) to limit dataset size for testing.
# Set to None for full run.
SAMPLE_SIZE = None

# =============================================================================
# Feature Extraction Hyperparameters
# =============================================================================

# Radial Distribution Function (RDF) Parameters
RDF_CUTOFF = 6.0  # Maximum distance in Angstroms
RDF_NUM_BINS = 60  # Number of histogram bins
RDF_SIGMA = 0.2  # Width of Gaussian smearing for smooth RDF

# Continuous Topological Moments (CTM) Parameters
CTM_BOND_CUTOFF = 3.0  # Cutoff distance (Angstroms) for defining bonded neighbors
CTM_ELEMENTS = ["Al", "Ga", "In", "O"]  # Elements to consider for chemical aggregation

# =============================================================================
# Model Hyperparameters (XGBoost)
# =============================================================================
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": 12,  # Utilize available vCPUs
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient histogram-based algorithm
    # Note: early_stopping_rounds is typically passed to fit(), not __init__
}

# Training Configuration
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
