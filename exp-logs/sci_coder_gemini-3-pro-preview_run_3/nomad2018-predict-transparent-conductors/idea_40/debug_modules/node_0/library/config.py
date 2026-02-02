import os

# ==========================================
# Global Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_40"
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission File Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# Feature Extraction Parameters
# ==========================================
# 1. Radial Distribution Function (RDF)
RDF_CUTOFF = 6.0  # Maximum distance in Angstroms
RDF_NUM_BINS = 60  # Number of bins for the histogram
RDF_SIGMA = 0.2  # Gaussian smearing width

# 2. Polyhedral Connectivity
# Cutoff distance to define a Metal-Oxygen bond for connectivity analysis
# This is critical for determining corner/edge/face sharing.
CONNECTIVITY_CUTOFF = 3.0  # Angstroms

# 3. Local Site Environment
# Cutoff for Bond Valence Sum (BVS) and Effective Coordination Number (ECoN)
LOCAL_ENV_CUTOFF = 6.0  # Angstroms

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
RANDOM_SEED = 42

# Configuration for the XGBoost Regressor
# Low learning rate and high estimators for better generalization
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.65,  # Row subsampling
    "colsample_bytree": 0.65,  # Column subsampling
    "min_child_weight": 1,
    "gamma": 0,
    "objective": "reg:squarederror",
    "n_jobs": -1,  # Use all available cores
    "random_state": RANDOM_SEED,
    "tree_method": "hist",  # Efficient histogram-based algorithm
}

# Target variables to predict
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# ==========================================
# Execution & Caching Control
# ==========================================
# Flag to enable debug mode (runs on a small subset of data)
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 100

# Flag to control caching behavior
# If True, the pipeline will attempt to load processed features from disk
LOAD_CACHED_DATA = True
