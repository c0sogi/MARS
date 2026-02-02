import os

# ==========================================
# Global Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Pre-split)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Feature Cache Paths
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Output Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Physical Constants & Feature Parameters
# ==========================================

# Bond Valence Sum (BVS) Parameters
# Formula: V_i = sum(exp((R0 - d_ij) / B))
# R0 values for Metal-Oxygen bonds (in Angstroms)
BVS_R0 = {"Al": 1.651, "Ga": 1.708, "In": 1.907}
BVS_B = 0.37  # Standard universal constant for BVS

# Radial Distribution Function (RDF) Settings
RDF_CUTOFF = 6.0  # Angstroms
RDF_BINS = 100  # Number of bins for the histogram
RDF_SIGMA = 0.2  # Gaussian smearing width (if applicable)

# Geometric Analysis Settings
# Cutoff distance to define neighbors for angle and coordination calculations
NEIGHBOR_CUTOFF = 3.0

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
RANDOM_SEED = 42

# XGBoost Regressor Parameters
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",  # Optimized for speed
    "importance_type": "gain",
}

# Training Control
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 200

# Target Variables
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
