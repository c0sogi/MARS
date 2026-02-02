import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Feature Cache Paths
# We use parquet for efficient storage of tabular feature data
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_combined_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_combined_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_combined_features.parquet")

# Submission Output
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Settings
# ==========================================
RANDOM_SEED = 42
# Set to a small integer (e.g., 100) to run a quick debug cycle on a subset of data.
# Set to None to run on the full dataset.
DEBUG_SAMPLE_SIZE = None

# ==========================================
# Feature Extraction Parameters
# ==========================================

# 1. Explicit Physical Descriptors
# (No specific hyperparameters, just calculating volume and density)

# 2. Radial Distribution Function (RDF)
# Moderate binning and reasonable cutoff to capture local geometry without sparsity
RDF_CUTOFF = 6.0  # Angstroms
RDF_BINS = 50  # Number of bins
ELEMENTS = ["Al", "Ga", "In", "O"]  # Elements expected in the structures

# 3. MatGL (Implicit Embeddings)
# Using the pre-trained M3GNet model for potential energy surfaces
MATGL_MODEL_NAME = "M3GNet-MP-2021.2.8-PES"

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
# Configuration for the XGBoost Regressor
# Low learning rate and high estimators for generalization
# Subsampling to leverage diverse feature views
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 1,
    "gamma": 0,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient histogram-based algorithm
}

# ==========================================
# Training Configuration
# ==========================================
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
LOG_TRANSFORM_TARGETS = True  # Apply log(1+y) to targets
EARLY_STOPPING_ROUNDS = 100  # Stop if validation score doesn't improve
VERBOSE_EVAL = 100  # Print evaluation every N rounds
