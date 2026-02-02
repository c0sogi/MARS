import os

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata CSV paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Cache directory for intermediate files (Features, Processed Data)
# Using 'idea_41' as the designated experiment identifier
CACHE_DIR = "./working/idea_41"
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission output path
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ==========================================
# Global Constants & Physics Parameters
# ==========================================
RANDOM_SEED = 42

# Atomic species present in the dataset
ATOMIC_SPECIES = ["Al", "Ga", "In", "O"]
METALS = ["Al", "Ga", "In"]
ANIONS = ["O"]

# Bond Valence Sum (BVS) Parameters
# R0 values for Metal(III)-Oxygen(-II) pairs
# Reference: Brown, I. D. & Altermatt, D. (1985). Acta Cryst. B41, 244-247.
BVS_PARAMS = {
    ("Al", "O"): 1.651,
    ("Ga", "O"): 1.708,
    ("In", "O"): 1.907,
    # Symmetric keys for lookups
    ("O", "Al"): 1.651,
    ("O", "Ga"): 1.708,
    ("O", "In"): 1.907,
}
BVS_B = 0.37  # Universal constant for BVS

# Geometric Cutoffs (in Angstroms)
# RDF_CUTOFF: Max distance for Radial Distribution Function
RDF_CUTOFF = 6.0
RDF_BIN_WIDTH = 0.1  # Width of bins for RDF histograms

# BONDING_CUTOFF: Max distance to consider atoms as bonded for
# angles, coordination number, and local anisotropy calculations.
# 3.0 A covers the first coordination shell for Al-O, Ga-O, In-O.
BONDING_CUTOFF = 3.0

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
# Optimized for generalization on small/medium datasets
XGB_PARAMS = {
    "n_estimators": 3000,  # High number of trees
    "learning_rate": 0.01,  # Low learning rate (shrinkage)
    "max_depth": 6,  # Moderate depth to prevent memorization
    "subsample": 0.7,  # Row subsampling
    "colsample_bytree": 0.6,  # Feature subsampling
    "objective": "reg:squarederror",
    "n_jobs": -1,  # Use all available cores
    "random_state": RANDOM_SEED,
    "tree_method": "hist",  # Efficient histogram-based algorithm
}

# Training control
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 200

# Debugging
# Set to a small integer (e.g., 100) to run pipeline on a subset of data.
# Set to None to run on the full dataset.
DEBUG_SAMPLE_SIZE = None
