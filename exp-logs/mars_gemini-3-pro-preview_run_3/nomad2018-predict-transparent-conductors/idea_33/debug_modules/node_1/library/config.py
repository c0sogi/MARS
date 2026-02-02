import os

# --- Directory Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_33")
SUBMISSION_DIR = "./submission"

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- File Paths ---
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Physical Constants ---
# Electronegativity (Pauling scale)
ELECTRONEGATIVITY = {"Al": 1.61, "Ga": 1.81, "In": 1.78, "O": 3.44}

# Bond Valence Parameters (Brown & Altermatt, 1985)
# R0 values for M-O bonds (Metal oxidation state +3, Oxygen -2)
BVS_PARAMS = {"R0": {"Al": 1.651, "Ga": 1.730, "In": 1.906}, "b": 0.37}

# --- Feature Extraction Hyperparameters ---
RDF_CUTOFF = 6.0  # Angstroms, max distance for Radial Distribution Function
RDF_BINS = 60  # Number of bins for RDF
BOND_CUTOFF = 3.0  # Angstroms, max distance to consider a bond for angles/BVS
PERCENTILES = [0, 10, 25, 50, 75, 90, 100]  # Percentiles for distribution aggregation

# --- Model Hyperparameters (XGBoost) ---
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient training
}

# --- Targets ---
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# --- Debugging ---
# Set to True to run on a small subset of data for testing pipeline
DEBUG = False
MAX_SAMPLES = 100 if DEBUG else None
