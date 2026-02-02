import os

# -----------------------------------------------------------------------------
# Global Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
# Raw Input
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Generated Metadata (Manifests)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Caching Paths for Intermediate Features
TRAIN_COMBINED_FEATURES_PATH = os.path.join(
    WORKING_DIR, "train_combined_features.parquet"
)
VAL_COMBINED_FEATURES_PATH = os.path.join(WORKING_DIR, "val_combined_features.parquet")
TEST_COMBINED_FEATURES_PATH = os.path.join(
    WORKING_DIR, "test_combined_features.parquet"
)

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# Atomic species to track for element-wise pooling
ATOMIC_SPECIES = ["Al", "Ga", "In", "O"]

# Target variables
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Tabular features available in the CSV/Metadata
TABULAR_FEATURES = [
    "spacegroup",
    "number_of_total_atoms",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "lattice_vector_1_ang",
    "lattice_vector_2_ang",
    "lattice_vector_3_ang",
    "lattice_angle_alpha_degree",
    "lattice_angle_beta_degree",
    "lattice_angle_gamma_degree",
]

# Explicit physical descriptors derived from geometry (to be calculated)
PHYSICAL_FEATURES = [
    "volume",
    "density",
    "num_atoms_geometry",  # Verification check against CSV
]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# XGBoost Regressor settings
# Strategy: Low learning rate, high estimators, stochastic subsampling to prevent overfitting
XGB_PARAMS = {
    "n_estimators": 2500,
    "learning_rate": 0.01,
    "max_depth": 7,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "min_child_weight": 5,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient training
}

# MatGL (M3GNet) Settings
# Using the standard pre-trained potential from Materials Project
MATGL_MODEL_NAME = "M3GNet-MP-2021.2.8-PES"
