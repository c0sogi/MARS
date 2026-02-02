import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
# Input data is read from the metadata directory which contains the processed splits
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"

# Working directory for caching intermediate files (Parquet/NPY)
# We use a specific subdirectory for this idea to avoid conflicts
WORKING_DIR = "./working/idea_8"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Absolute paths to key files
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATASET COLUMNS
# =============================================================================
ID_COL = "Id"
TARGET_COL = "Cover_Type"

# Feature groups for Reverse One-Hot Encoding
# Soil Types are Soil_Type1 to Soil_Type40
SOIL_COLUMNS = [f"Soil_Type{i}" for i in range(1, 41)]

# Wilderness Areas are Wilderness_Area1 to Wilderness_Area4
WILDERNESS_COLUMNS = [f"Wilderness_Area{i}" for i in range(1, 5)]

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
N_FOLDS = 5

# The dataset typically covers classes 1-7.
# We set this to 7 to handle the potential range, though mapping will occur in the pipeline.
NUM_CLASSES = 7

# Debugging: Set to an integer (e.g., 10000) to subsample data for rapid prototyping.
# Set to None to use the full dataset (Required for final submission).
DEBUG_SAMPLE_SIZE = None

# Feature Engineering Flags
USE_GEOMETRIC_FEATURES = True
USE_REVERSE_ONE_HOT = True

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# --- Level 0: Base Learners (XGBoost) ---
# High capacity models trained on raw features + engineered features
L0_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,
    "tree_method": "hist",  # Optimized for GPU
    "device": "cuda",  # Use NVIDIA A100
    "max_depth": 10,  # High capacity as per strategy
    "learning_rate": 0.05,  # Robust learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "merror",
    "seed": SEED,
    "n_jobs": 12,
    "verbosity": 0,
}

# Training rounds for Level 0
L0_NUM_BOOST_ROUND = 3000
L0_EARLY_STOPPING_ROUNDS = 50
