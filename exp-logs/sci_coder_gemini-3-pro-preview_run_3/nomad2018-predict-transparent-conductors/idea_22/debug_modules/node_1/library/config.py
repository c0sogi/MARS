import os

# -----------------------------------------------------------------------------
# Directories and File Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
# Cache directory for intermediate files (Idea 22)
WORK_DIR = "./working/idea_22"
SUBMISSION_DIR = "./submission"
METADATA_DIR = "./metadata"

# Create necessary directories
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Raw Data Paths
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# -----------------------------------------------------------------------------
# Global Constants
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
# Elements present in the dataset
ATOM_LIST = ["Al", "Ga", "In", "O"]

# -----------------------------------------------------------------------------
# Feature Extraction Hyperparameters
# -----------------------------------------------------------------------------
# Cutoff radius for defining bonded neighbors (e.g. for coordination number, bond angles)
BOND_CUTOFF = 3.0  # Angstroms

# Cutoff radius for Radial Distribution Function (RDF)
RDF_CUTOFF = 6.0  # Angstroms

# Number of bins for RDF histograms
RDF_NUM_BINS = 60

# Percentiles to compute for distributional features (Cation & Anion sublattices)
# Captures Min, 25%, Median, 75%, Max of local property distributions
PERCENTILES = [0, 25, 50, 75, 100]

# -----------------------------------------------------------------------------
# Model Hyperparameters (XGBoost)
# -----------------------------------------------------------------------------
# Parameters optimized for generalization on this specific task
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient training
}

# Early stopping rounds to prevent overfitting
EARLY_STOPPING_ROUNDS = 100

# -----------------------------------------------------------------------------
# Target Variables
# -----------------------------------------------------------------------------
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
