import os

# ==========================================
# Global Configuration and Constants
# ==========================================

# Random Seed for Reproducibility
RANDOM_SEED = 42

# ==========================================
# Directory Paths
# ==========================================

# Input Data Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working Directory for Caching Intermediate Files (Idea 5)
WORKING_DIR = "./working/idea_5"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata File Paths (Generated previously)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Submission File
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Physical Constants
# ==========================================

# Atomic Masses (u) for Density Calculation
# Values taken from standard periodic table
ATOMIC_MASSES = {"Al": 26.9815385, "Ga": 69.723, "In": 114.818, "O": 15.999}

# ==========================================
# Model Hyperparameters
# ==========================================

# MatGL Configuration
# Using a pre-trained potential model available in matgl
MATGL_MODEL_NAME = "M3GNet-MP-2021.2.8-PES"

# XGBoost Regressor Hyperparameters
# Strategy: Low learning rate (shrinkage) and high estimators for generalization.
# Subsampling (row and column) to prevent overfitting to high-dim embeddings.
XGB_PARAMS = {
    "n_estimators": 2500,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Faster training
    "early_stopping_rounds": 50,
}

# Target Column Names
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
