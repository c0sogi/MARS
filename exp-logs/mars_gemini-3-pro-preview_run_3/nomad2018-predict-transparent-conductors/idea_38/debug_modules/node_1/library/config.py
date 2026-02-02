import os

# ==========================================
# 1. File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_38"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cached feature file paths
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# ==========================================
# 2. Random Seed
# ==========================================
RANDOM_SEED = 42

# ==========================================
# 3. Feature Extraction Parameters
# ==========================================

# --- Radial Distribution Function (RDF) ---
RDF_CUTOFF = 6.0  # Angstroms
RDF_NUM_BINS = 60  # Number of bins for the histogram
RDF_SIGMA = 0.2  # Width for Gaussian smearing (if used instead of raw histogram)

# --- Local Environment (BVS, ECoN, Anisotropy) ---
# Cutoff distance to define the first coordination shell for local descriptors
BOND_CUTOFF = 3.0  # Angstroms

# Bond Valence Sum (BVS) Parameters
# R0 values for Metal-Oxygen bonds (Al-O, Ga-O, In-O)
# Formula: exp((R0 - d) / b)
BVS_PARAMS = {"Al": 1.62, "Ga": 1.73, "In": 1.92, "b": 0.37}

# Elements to consider for resolved features
CATIONS = ["Al", "Ga", "In"]
ANIONS = ["O"]
ALL_ELEMENTS = CATIONS + ANIONS

# Percentiles for aggregating local distributions
PERCENTILES = [0, 25, 50, 75, 100]

# ==========================================
# 4. Model Hyperparameters (XGBoost)
# ==========================================
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient for larger datasets
}

# Target column names
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
