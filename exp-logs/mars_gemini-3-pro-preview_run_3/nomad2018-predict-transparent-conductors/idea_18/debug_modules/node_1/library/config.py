import os

# =============================================================================
# GLOBAL CONFIGURATION & REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_18")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Data Files
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Generated Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Cache Files for Processed Features (Parquet format)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Output Submission File
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# PHYSICAL CONSTANTS & SUBLATTICE DEFINITIONS
# =============================================================================
# Atomic numbers for mapping species
ATOMIC_NUMBERS = {"Al": 13, "Ga": 31, "In": 49, "O": 8}

# Sublattice definitions for hierarchical grouping
SUBLATTICE_METALS = ["Al", "Ga", "In"]
SUBLATTICE_ANIONS = ["O"]

# Target column names
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# =============================================================================
# FEATURE EXTRACTION HYPERPARAMETERS
# =============================================================================
# Radial Distribution Function (RDF) settings
RDF_CUTOFF = 6.0  # Angstroms: Maximum distance for pair correlations
RDF_BINS = 60  # Number of bins for the histogram (resolution = 0.1 A)
RDF_SIGMA = 0.2  # Width for Gaussian smearing (if applicable)

# Local Geometric Moments settings
NEIGHBOR_CUTOFF = 3.0  # Angstroms: Cutoff to define nearest neighbors for bond stats

# =============================================================================
# MODEL HYPERPARAMETERS (XGBoost)
# =============================================================================
# Optimized for robustness against noise and feature dilution
XGB_PARAMS = {
    "n_estimators": 3000,  # High number of trees
    "learning_rate": 0.01,  # Low learning rate for shrinkage
    "max_depth": 6,  # Constrain depth to prevent overfitting
    "subsample": 0.7,  # Stochastic row subsampling
    "colsample_bytree": 0.7,  # Stochastic column subsampling
    "n_jobs": -1,  # Use all available cores
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient histogram-based algorithm
}
