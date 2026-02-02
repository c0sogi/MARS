import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_36"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache File Paths
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# ==========================================
# Global Settings
# ==========================================
RANDOM_SEED = 42
N_JOBS = 12  # Number of CPU cores available

# Debugging
# Set DEBUG to True to use a small subset of data for testing the pipeline
DEBUG = False
DEBUG_SAMPLE_SIZE = 100

# ==========================================
# Feature Extraction Hyperparameters
# ==========================================

# 1. Radial Distribution Function (RDF)
RDF_CUTOFF = 6.0  # Angstroms
RDF_BINS = 60  # Number of bins
RDF_SIGMA = 0.2  # Gaussian smearing width (if applicable)

# 2. Bond Valence Sum (BVS)
# R0 parameters for Metal-Oxygen bonds (Brown & Altermatt, 1985)
# B is typically 0.37
BVS_PARAMS = {
    "Al": {"R0": 1.651, "B": 0.37},
    "Ga": {"R0": 1.708, "B": 0.37},
    "In": {"R0": 1.907, "B": 0.37},
    "O": {"R0": 0.0, "B": 0.37},  # Placeholder
}

# 3. Coordination & Topology
# Cutoff distance to define a "bond" for angle calculations
BOND_CUTOFF = 3.0  # Angstroms (covers first coordination shell for Al/Ga/In-O)
# Cutoff distance for Effective Coordination Number (ECoN)
ECON_CUTOFF = 6.0  # Angstroms

# 4. Distributional Statistics
# Percentiles to compute for BVS, ECoN, and Angle distributions
PERCENTILES = [0, 25, 50, 75, 100]

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 7,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "n_jobs": N_JOBS,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient for larger datasets
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
}

# Target Columns
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
