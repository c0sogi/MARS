import os

# -----------------------------------------------------------------------------
# Global Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Domain Knowledge: Atomic Properties
# -----------------------------------------------------------------------------
# Mapping of atomic species to physical properties.
# Properties:
#   - Electronegativity: Pauling scale
#   - Atomic Radius: Empirical/Covalent (Angstroms) - specific to oxide contexts
#   - Valence Electrons: Group number (O=6, Al/Ga/In=3)
ATOMIC_PROPERTIES = {
    "Al": {"electronegativity": 1.61, "radius": 1.43, "valence": 3},
    "Ga": {"electronegativity": 1.81, "radius": 1.35, "valence": 3},
    "In": {"electronegativity": 1.78, "radius": 1.67, "valence": 3},
    "O": {"electronegativity": 3.44, "radius": 0.73, "valence": 6},
}

# -----------------------------------------------------------------------------
# Feature Engineering Configuration
# -----------------------------------------------------------------------------
# Cutoff distance (Angstroms) for determining neighbors in bond statistics
NEIGHBOR_CUTOFF = 3.0

# -----------------------------------------------------------------------------
# Model Hyperparameters (XGBoost)
# -----------------------------------------------------------------------------
# Configuration for the XGBoost regressor.
# Tuned for generalization on small-to-medium datasets with non-linear features.
XGB_PARAMS = {
    "n_estimators": 2500,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 1,
    "gamma": 0.1,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",  # Efficient training
}
