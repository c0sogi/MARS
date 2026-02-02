import os

# =============================================================================
# Directories and File Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Cached Feature Files (Parquet)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Output Submission
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Constants
# =============================================================================
RANDOM_SEED = 42
ATOM_TYPES = ["Al", "Ga", "In", "O"]
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# =============================================================================
# Feature Extraction Hyperparameters
# =============================================================================

# Radial Distribution Function (RDF) Parameters
RDF_CUTOFF = 6.0  # Angstroms: Maximum distance for pairwise interactions
RDF_NUM_BINS = 60  # Number of bins for the histogram
RDF_SIGMA = 0.1  # Width for Gaussian smearing (if used)

# Voronoi Tessellation Parameters
# Supercell repetition to ensure correct Voronoi cells for atoms near boundaries
VORONOI_SUPERCELL_REPEAT = (3, 3, 3)

# =============================================================================
# Model Hyperparameters (XGBoost)
# =============================================================================
# Optimized for generalization on dense structural features
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 7,
    "subsample": 0.65,  # Row subsampling to prevent overfitting
    "colsample_bytree": 0.65,  # Column subsampling
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",  # Efficient training method
    "early_stopping_rounds": 100,
    "verbosity": 0,
}
