import os

# --- File Paths ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# --- Data Constants ---
RANDOM_SEED = 42
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# --- Geometric Feature Extraction Hyperparameters ---
# Cutoff distance for Radial Distribution Functions (RDF) in Angstroms
RDF_CUTOFF = 6.0
# Number of bins for RDF histograms
RDF_NUM_BINS = 60

# Cutoff distance for defining bonded neighbors (for bond lengths/angles) in Angstroms
BOND_CUTOFF = 3.0

# Percentiles to compute for interaction distributions (Min, 10%, 25%, Median, 75%, 90%, Max)
PERCENTILES = [0, 10, 25, 50, 75, 90, 100]

# --- Model Hyperparameters (XGBoost) ---
# Optimized for generalization with shrinkage and stochastic subsampling
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Faster training
}
