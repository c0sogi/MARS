import os

# --- Directory Paths ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_34"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- File Paths ---
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# --- Global Settings ---
RANDOM_SEED = 42

# --- Atomic Parameters for Feature Engineering ---
# Bond Valence Sum (BVS) R0 parameters for Metal-Oxygen pairs
# Values typically derived from Brown & Altermatt (1985) or similar databases
# Units: Angstroms
R0_VALUES = {"Al": 1.651, "Ga": 1.708, "In": 1.907}

# Oxidation states for BVS calculation (Target valence)
# Used to calculate BVS discrepancy if needed, though raw BVS is often used as feature
OXIDATION_STATES = {"Al": 3, "Ga": 3, "In": 3, "O": -2}

# --- Feature Engineering Hyperparameters ---
# Percentiles for aggregating atomic-level features (BVS, ECoN, Angles) into structure-level vectors
PERCENTILES = [0, 25, 50, 75, 100]

# RDF (Radial Distribution Function) Settings
RDF_R_MIN = 0.0
RDF_R_MAX = 10.0  # Angstroms, sufficient to capture medium-range order
RDF_N_BINS = 100  # Number of bins for the histogram
RDF_SIGMA = 0.2  # Width of Gaussian smearing for RDF (if continuous RDF is used)

# Angle distribution settings
ANGLE_N_BINS = 50
ANGLE_MIN = 0.0
ANGLE_MAX = 180.0

# --- Model Hyperparameters (XGBoost) ---
# Optimized for generalization on this specific materials dataset
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Faster training
}

# --- Target Configuration ---
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
