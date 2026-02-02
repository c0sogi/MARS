import os

# --- Paths ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Cached Feature File Paths
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Submission Path
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# --- Feature Extraction Hyperparameters ---
# Elements present in the dataset
ELEMENTS = ["Al", "Ga", "In", "O"]

# Radial Distribution Function (RDF) Parameters
RDF_CUTOFF = 6.0  # Angstroms
RDF_BINS = 60  # Number of bins for the histogram (resolution ~0.1 A)

# Local Environment Parameters
NEIGHBOR_CUTOFF = 3.0  # Angstroms, for defining bonded neighbors
PERCENTILES = [0, 25, 50, 75, 100]  # Percentiles for distributional aggregation

# --- Model Hyperparameters (XGBoost) ---
# Common parameters for both targets
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": 42,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Faster training
}

# Specific overrides if needed (currently using same for both)
FORMATION_ENERGY_PARAMS = XGB_PARAMS.copy()
BANDGAP_ENERGY_PARAMS = XGB_PARAMS.copy()

# Target Column Names
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Random Seed
RANDOM_SEED = 42
